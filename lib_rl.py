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

RLConfig = collections.namedtuple(
    "RLConfig",
    [
        "discount_factor",
        "n_epochs",
        "update_value_model_every_n_episodes",
        "batch_size",
        "lr",
        "replay_buffer_size",
    ],
)


class Game(object):
    def __init__(
        self,
        game_config: GameConfig,
        rl_config: RLConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        seed: Optional[int] = None,
    ):
        self.game_config = game_config
        self.rl_config = rl_config
        self.replay_buffer = ReplayBuffer(max_size=rl_config.replay_buffer_size)
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=rl_config.lr)
        self.env = lib_hanabi.HanabiEnvironment(game_config, seed=seed)
        self.value_model = deepcopy(self.model)
        self.epsilon_greedy = EpsilonGreedy()

    @torch.no_grad()
    def _get_best_action(self, state: GameState):
        X = lib_one_hot.encode(game_config=self.game_config, game_state=state)
        reward_per_action: torch.Tensor = self.model.forward(X)
        best_score, best_action_index = reward_per_action.max(axis=-1)
        best_action = action_of_index(
            best_action_index, hand_size=self.game_config.hand_size
        )
        return (best_action, best_score)

    def _step(self, state: GameState) -> Tuple[int, bool, GameState]:
        if self.epsilon_greedy.flip():
            action_source = "rand"
            action = self.env.sample_action()
            expected_score = self.model.forward(
                lib_one_hot.encode(game_config=self.game_config, game_state=state)
            )[index_of_action(action, hand_size=self.game_config.hand_size)]
        else:
            action_source = "best"
            action, expected_score = self._get_best_action(state)

        formatted_action = pretty_print_action(action, state)

        (state_new, reward, done) = self.env.step(action)

        compact_logger.print(
            f"Action {action_source}-{formatted_action} "
            f"for expected reward {expected_score: 2.2f}; "
            f"realized {reward: 2.2f}."
        )

        action_encoded = torch.tensor(
            index_of_action(action, self.game_config.hand_size)
        ).long()
        t = (
            lib_one_hot.encode(game_config=self.game_config, game_state=state),
            action_encoded,
            as_torch(reward),
            lib_one_hot.encode(game_config=self.game_config, game_state=state_new),
            as_torch(done),
        )
        self.replay_buffer.push(t)
        points = 0
        if reward == 1.0:
            points = 1
        return points, done, state_new

    @torch.no_grad()
    def play_one_episode(self):
        self.epsilon_greedy.on_epoch_start()
        state = self.env.reset()
        total_points = 0
        done = False
        while not done:
            points, done, state = self._step(state)
            total_points += points

        compact_logger.print(f"Got total points {total_points}.")

    def train(self, n_batches: int):
        with torch.no_grad():
            samples = self.replay_buffer.sample(self.rl_config.batch_size * n_batches)
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

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, action, one_step_lookahead),
            batch_size=self.rl_config.batch_size,
        )

        losses = []
        for (X, action, one_step_lookahead) in data_loader:
            self.optimizer.zero_grad()
            predicted = torch.gather(
                self.model(X), -1, torch.unsqueeze(action, -1)
            ).float()
            loss = F.smooth_l1_loss(predicted, one_step_lookahead)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))
        return np.mean(losses)

    def update_value_model(self):
        self.value_model = deepcopy(self.model)
        _freeze_model(self.value_model)


def _freeze_model(model):
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False


def pretty_print_action(action, state: GameState):
    (action_type, card_index) = action
    card = card_to_str(state.hands[0][card_index])
    if action_type == ActionType.PLAY_CARD:
        return f"P({card})"
    else:
        return f"D({card})"
