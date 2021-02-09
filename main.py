from lib_util import compact_logger
from lib_types import GameConfig
import warnings
import lib_rl

warnings.filterwarnings("ignore")


GAME_CONFIG = GameConfig(
    n_suits=1,
    n_ranks=5,
    n_copies=[3, 3, 3, 3, 3],
    n_players=1,
    n_max_hints=7,
    hand_size=5,
    max_lives=3,
)

RL_CONFIG = lib_rl.RLConfig(
    discount_factor=1.0,
    n_epochs=100_000,
    update_value_model_every_n_episodes=100,
    batch_size=128,
    lr=0.01,
)


def main():
    game = lib_rl.Game(game_config=GAME_CONFIG, rl_config=RL_CONFIG)
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

        avg_loss = game.train(n_batches=20)
        compact_logger.print(f"Average loss: {avg_loss: .3f}.")


if __name__ == "__main__":
    main()
