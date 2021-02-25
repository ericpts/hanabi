from lib_agent import ReplayBuffer, EpsilonGreedy
from typing import Tuple
from typing import Optional
import lib_one_hot
from lib_util import as_torch, compact_logger, card_to_str
from lib_types import (
    ActionType,
    action_of_index,
    GameConfig,
    GameState,
    index_of_action,
)
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim
import lib_hanabi
import collections
import torch.nn as nn
import gym

RUNNING_METRIC_DECAY = 0.99

RLConfig = collections.namedtuple(
    "RLConfig",
    [
        "discount_factor",
        "n_epochs",
        "update_value_model_every_n_steps",
        "batch_size",
        "lr",
        "replay_buffer_size",
        "optimizer",
        "minimum_replay_buffer_size_for_training",
    ],
)


class RLGame(object):
    def __init__(
        self,
        rl_config: RLConfig,
        model: nn.Module,
        env: gym.Env,
    ):
        self.rl_config = rl_config
        self.replay_buffer = ReplayBuffer(max_size=rl_config.replay_buffer_size)
        self.model = model
        self.optimizer = rl_config.optimizer(self.model.parameters(), lr=rl_config.lr)
        self.env = env
        self.value_model = deepcopy(self.model)
        self.epsilon_greedy = EpsilonGreedy()
        self.running_loss = 0.0
        self.running_points = 0.0
        self.total_steps = 0

    def run(self):
        for epoch in range(self.rl_config.n_epochs):
            compact_logger.on_episode_start()
            compact_logger.print(
                f"Running epoch {epoch}; "
                f"eps greedy is {self.epsilon_greedy.get():.2f}"
            )
            self._play_one_episode()

    def _play_one_episode(self):
        self.epsilon_greedy.on_epoch_start()
        state = self.env.reset()
        done = False
        while not done:
            points, done, state = self._step(state)
            self.total_steps += 1
            self.running_points = (
                RUNNING_METRIC_DECAY * self.running_points
                + (1 - RUNNING_METRIC_DECAY) * points
            )

            if (
                len(self.replay_buffer)
                < self.rl_config.minimum_replay_buffer_size_for_training
            ):
                continue

            if self.total_steps % self.rl_config.update_value_model_every_n_steps == 0:
                self._update_value_model()

            self.running_loss = (
                RUNNING_METRIC_DECAY * self.running_loss
                + (1 - RUNNING_METRIC_DECAY) * self._train_once()
            )

        compact_logger.print(f"Running points: {self.running_points:.2f}.")
        compact_logger.print(f"Running loss: {self.running_loss:.2f}.")

    @torch.no_grad()
    def _step(self, state: np.ndarray) -> Tuple[int, bool, GameState]:
        if self.epsilon_greedy.flip():
            action_source = "rand"
            action = torch.as_tensor(self.env.action_space.sample())
            expected_score = self.model.forward(state)[action]
        else:
            action_source = "best"
            expected_score, action = self._get_best_action(state)

        (state_new, reward, done) = self.env.step(action)

        compact_logger.print(
            f"Action {action_source}-{action} "
            f"for expected reward {expected_score: 2.2f}; "
            f"realized {reward: 2.2f}."
        )

        t = (
            state,
            action,
            as_torch(reward),
            state_new,
            as_torch(done),
        )
        self.replay_buffer.push(t)
        points = 0
        if reward == 1.0:
            points = 1
        return points, done, state_new

    @torch.no_grad()
    def _get_best_action(self, state: np.ndarray) -> Tuple[int, float]:
        reward_per_action: torch.Tensor = self.model.forward(state)
        best_score, best_action_index = reward_per_action.max(axis=-1)
        return best_score, best_action_index

    def _train_once(self) -> float:
        with torch.no_grad():
            samples = self.replay_buffer.sample(self.rl_config.batch_size)
            (X, action, reward, X_new, done) = map(torch.stack, zip(*samples))
            value_after_action = self.value_model.forward(X_new).max(axis=-1).values
            add_value_after_action = 1 - done
            one_step_lookahead = (
                reward
                + add_value_after_action
                * self.rl_config.discount_factor
                * value_after_action
            )
            one_step_lookahead = torch.unsqueeze(one_step_lookahead, -1)

        self.optimizer.zero_grad()
        predicted = torch.gather(self.model(X), -1, torch.unsqueeze(action, -1)).float()
        loss = F.smooth_l1_loss(predicted, one_step_lookahead)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _update_value_model(self):
        self.value_model = deepcopy(self.model)
        _freeze_model(self.value_model)


def _freeze_model(model):
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False
