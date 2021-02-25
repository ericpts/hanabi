from lib_util import compact_logger
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
    discount_factor=1.0,
    n_epochs=20_000,
    update_value_model_every_n_episodes=20,
    batch_size=32,
    lr=0.001,
    replay_buffer_size=1_000,
    optimizer=torch.optim.Adam,
)


def main():
    env = lib_hanabi.HanabiEnvironment(GAME_CONFIG)
    game = lib_rl.Game(
        rl_config=RL_CONFIG,
        model=lib_agent.SimpleModel(
            input_size=np.prod(env.observation_space.shape),
            output_size=env.action_space.n,
            fc_sizes=[100, 100],
        ),
        env=env,
    )
    for epoch in range(RL_CONFIG.n_epochs):
        compact_logger.on_episode_start()
        compact_logger.print(
            f"Running epoch {epoch}; " f"eps greedy is {game.epsilon_greedy.get():.2f}"
        )
        game.play_one_episode()

        if len(game.replay_buffer) < 1_000:
            continue

        if epoch % RL_CONFIG.update_value_model_every_n_episodes == 0:
            game.update_value_model()

        avg_loss = game.train(n_batches=10)
        compact_logger.print(f"Average loss: {avg_loss: .3f}.")


if __name__ == "__main__":
    main()
