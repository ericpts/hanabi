import numpy as np
import torch
import torch.optim
from lib_types import GameConfig
import warnings
import lib_rl
import lib_agent
import lib_hanabi

warnings.filterwarnings("ignore")


GAME_CONFIG = GameConfig(
    n_suits=4,
    n_ranks=5,
    n_copies=[3, 3, 3, 3, 3],
    n_players=1,
    n_max_hints=7,
    hand_size=5,
    max_lives=3,
)

RL_CONFIG = lib_rl.RLConfig(
    discount_factor=0.9,
    n_epochs=20_000,
    update_value_model_every_n_steps=500,
    batch_size=8,
    lr=0.001,
    replay_buffer_size=1000,
    optimizer=torch.optim.Adam,
    minimum_replay_buffer_size_for_training=1000,
)


def main():
    env = lib_hanabi.HanabiEnvironment(GAME_CONFIG)
    lib_rl.RLGame(
        rl_config=RL_CONFIG,
        model=lib_agent.SimpleModel(
            input_size=np.prod(env.observation_space.shape),
            output_size=env.action_space.n,
            fc_sizes=[100, 100, 100, 100, 100],
        ),
        env=env,
    ).run()


if __name__ == "__main__":
    main()
