from lib_agent import SimpleModel, ReplayBuffer, EpsilonGreedy
import lib_one_hot
from lib_util import as_torch
from lib_types import (
    ActionType,
    action_of_index,
    GameConfig,
    GameState,
    index_of_action,
)
from contextlib import contextmanager
import time
from copy import deepcopy
import numpy as np
import string
from enum import Enum
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim
import lib_hanabi
import warnings

warnings.filterwarnings("ignore")


class CompactLogger(object):
    def __init__(self, log_every_n: int = 10):
        self.n_lines = 0
        self.log_every_n = log_every_n
        self.at = 0

    def on_episode_start(self):
        self.at += 1
        if self.at == self.log_every_n:
            self.at = 0
            self.reset()

    def print(self, message: str):
        if self.at != 0:
            return
        self.n_lines += 1
        print(message)

    def reset(self):
        for _ in range(self.n_lines):
            print("\033[F" + (" " * 80), end="")
        print("\r", end="")
        self.n_lines = 0


compact_logger = CompactLogger()

GAME_CONFIG = GameConfig(
    n_suits=4,
    n_ranks=5,
    n_copies=[3, 2, 2, 2, 1],
    n_players=1,
    n_max_hints=7,
    hand_size=5,
    max_lives=3,
)

DISCOUNT_FACTOR = 1.0
N_EPOCHS = 200_000
UPDATE_VALUE_MODEL_EVERY_N_EPISODES = 10
BATCH_SIZE = 64


def freeze_model(model):
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False


def card_to_str(card):
    alphabet = string.ascii_uppercase
    (suit_index, number) = card
    suit = alphabet[suit_index]
    return f"{suit}{number}"


class Game(object):
    def __init__(self, game_config: GameConfig):
        self.game_config = game_config
        self.replay_buffer = ReplayBuffer()
        self.model = SimpleModel(self.game_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.env = lib_hanabi.HanabiEnvironment(GAME_CONFIG, seed=1)
        self.value_model = deepcopy(self.model)
        self.epsilon_greedy = EpsilonGreedy()

    def _get_best_action(self, state: GameState):
        X = torch.from_numpy(
            lib_one_hot.encode(game_config=self.game_config, game_state=state)
        ).float()
        valid_actions = self.model.forward(X)
        best_action = action_of_index(
            int(valid_actions.argmax(axis=-1)), hand_size=self.game_config.hand_size
        )
        best_score = valid_actions.max(axis=-1).values
        return (best_action, best_score)

    def pretty_print_action(self, action, state: GameState):
        (action_type, card_index) = action
        card = card_to_str(state.hands[0][card_index])
        if action_type == ActionType.PLAY_CARD:
            return f"P({card})"
        else:
            return f"D({card})"

    @torch.no_grad()
    def play_one_episode(self):
        self.epsilon_greedy.on_epoch_start()
        state = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            if self.epsilon_greedy.flip():
                action = self.env.sample_action()
                expected_score = self.model.forward(
                    torch.from_numpy(
                        lib_one_hot.encode(
                            game_config=self.game_config, game_state=state
                        )
                    ).float()
                )[index_of_action(action, hand_size=self.game_config.hand_size)]
            else:
                action, expected_score = self._get_best_action(state)

            compact_logger.print(
                f"Action {self.pretty_print_action(action, state)} "
                f"for expected reward {expected_score:.2f}"
            )

            (state_new, reward, done) = self.env.step(action)
            action_encoded = torch.tensor(
                index_of_action(action, self.game_config.hand_size)
            ).long()
            t = (
                as_torch(
                    lib_one_hot.encode(game_config=self.game_config, game_state=state)
                ),
                action_encoded,
                as_torch(reward),
                as_torch(
                    lib_one_hot.encode(
                        game_config=self.game_config, game_state=state_new
                    )
                ),
                as_torch(done),
            )
            self.replay_buffer.push(t)
            state = state_new
            total_reward += reward

        compact_logger.print(f"Got total reward {total_reward:.2f}.")

    def train(self, n_batches: int):
        with torch.no_grad():
            samples = self.replay_buffer.sample(BATCH_SIZE * n_batches)
            (X, action, reward, X_new, done) = map(torch.stack, zip(*samples))
            value_after_action = self.value_model.forward(X_new).max(axis=-1).values
            add_value_after_action = 1 - done
            one_step_lookahead = (
                reward + add_value_after_action * DISCOUNT_FACTOR * value_after_action
            )
            one_step_lookahead = torch.unsqueeze(one_step_lookahead, -1)

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, action, one_step_lookahead),
            batch_size=BATCH_SIZE,
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
        freeze_model(self.value_model)


def main():
    global n_lines
    game = Game(game_config=GAME_CONFIG)
    for epoch in range(N_EPOCHS):
        compact_logger.on_episode_start()
        game.play_one_episode()

        if len(game.replay_buffer) < 1_000:
            continue

        if epoch % UPDATE_VALUE_MODEL_EVERY_N_EPISODES == 0:
            game.update_value_model()

        avg_loss = game.train(n_batches=128)
        compact_logger.print(f"Average loss: {avg_loss: .3f}.")


if __name__ == "__main__":
    main()
