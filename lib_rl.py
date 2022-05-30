from lib_agent import ReplayBuffer, EpsilonGreedy
from typing import Tuple
from typing import Optional
import lib_one_hot
from lib_util import as_torch, compact_logger, card_to_str, AverageAccumulator
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
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format


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
        env: lib_hanabi.HanabiEnvironment,
    ):
        print(f"Using model: {model}")
        self.rl_config = rl_config
        self.replay_buffer = ReplayBuffer(max_size=rl_config.replay_buffer_size)
        self.model = model
        self.optimizer = rl_config.optimizer(self.model.parameters(), lr=rl_config.lr)
        self.env = env
        self.value_model = deepcopy(self.model)
        self.epsilon_greedy = EpsilonGreedy()
        self.total_steps = 0

    def run(self):
        for epoch in range(self.rl_config.n_epochs):
            compact_logger.on_episode_start()
            compact_logger.print("\n" * 5)
            compact_logger.print("=" * 80)
            compact_logger.print(
                f"Running epoch {epoch}; "
                f"eps greedy is {self.epsilon_greedy.get():.2f}"
            )
            self._play_one_episode()

    def _play_one_episode(self):
        episode_loss = AverageAccumulator("loss")
        episode_points = 0

        self.epsilon_greedy.on_epoch_start()
        state = self.env.reset()
        done = False

        while not done:
            points, done, state = self._step(state)
            self.total_steps += 1
            episode_points += points

            if (
                len(self.replay_buffer)
                < self.rl_config.minimum_replay_buffer_size_for_training
            ):
                continue

            if self.total_steps % self.rl_config.update_value_model_every_n_steps == 0:
                self._update_value_model()

            loss = self._train_once()
            episode_loss.add(loss)

        compact_logger.print(f"Episode points: {episode_points}.")
        compact_logger.print(f"Episode loss: {episode_loss.average():.2f}.")

    @torch.no_grad()
    def _step(self, state: np.ndarray) -> Tuple[int, bool, GameState]:
        if self.epsilon_greedy.flip():
            action_source = "rand"
            action = torch.as_tensor(self.env.action_space.sample())
            expected_score = self.model.forward(state)[action]
        else:
            action_source = "best"
            expected_score, action = self._get_best_action(state)

        compact_logger.print(f"Environment: {self.env.pretty_print()}")
        compact_logger.print(f"Expected score: ")

        score_per_action_index = self.model.forward(state)

        rows = []
        for action_type in [ActionType.PLAY_CARD, ActionType.DISCARD]:
            current_row = []
            for card_index in range(self.env.game_config.hand_size):
                action_index = index_of_action(
                    (action_type, card_index), self.env.game_config.hand_size
                )
                score = float(score_per_action_index[action_index])

                current_row.append(score)

            rows.append(current_row)

        if compact_logger.is_printing():
            df = pd.DataFrame(
                rows,
                ["PLAY", "DISC"],
                [
                    self.env._pretty_print_hand(card_index)
                    for card_index in range(self.env.game_config.hand_size)
                ],
            )
            compact_logger.print(df)

            action_type, card_index = action_of_index(
                action, self.env.game_config.hand_size
            )
            ppat = {
                ActionType.PLAY_CARD: "PLAY",
                ActionType.DISCARD: "DISC",
            }[action_type]
            pp_card = self.env._pretty_print_hand(card_index)

            compact_logger.print(
                f"Action {action_source}-{ppat}({pp_card}) "
                f"for expected reward {expected_score: 2.2f}; "
            )

        (state_new, reward, done) = self.env.step(action)

        compact_logger.print(f"Realized {reward: 2.2f}.")

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
        loss = F.mse_loss(predicted, one_step_lookahead)
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
