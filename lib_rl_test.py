import lib_one_hot
import lib_agent
import torch
import lib_types
import lib_rl
from lib_types import index_of_action
import pytest


@torch.no_grad()
@pytest.mark.randomize(seed=int)
def test_game_get_best_action_simple(seed):
    game_config = lib_types.GameConfig(
        n_suits=1,
        n_ranks=5,
        n_copies=[3, 3, 3, 3, 3],
        n_players=1,
        n_max_hints=7,
        hand_size=5,
        max_lives=3,
    )

    rl_config = lib_rl.RLConfig(
        discount_factor=1.0,
        n_epochs=100_000,
        update_value_model_every_n_episodes=100,
        batch_size=128,
        lr=0.0001,
        replay_buffer_size=1,
    )

    torch.manual_seed(seed)
    game = lib_rl.Game(
        game_config, rl_config, seed=seed, model=lib_agent.SimpleModel(game_config, [])
    )

    state = game.env.reset()

    expected_value, expected_index = game.model.forward(
        lib_one_hot.encode(game_config=game_config, game_state=state)
    ).max(axis=-1)

    actual_action, actual_value = game._get_best_action(state)

    assert actual_value == expected_value
    assert (
        index_of_action(actual_action, hand_size=game_config.hand_size)
        == expected_index
    )


@torch.no_grad()
@pytest.mark.randomize(
    seed=int, n_suits=int, n_ranks=int, hand_size=int, min_num=1, max_num=100
)
def test_game_get_best_action_and_playing(
    seed, n_suits: int, n_ranks: int, hand_size: int
):
    game_config = lib_types.GameConfig(
        n_suits=n_suits,
        n_ranks=n_ranks,
        n_copies=[3] * n_ranks,
        n_players=1,
        n_max_hints=7,
        hand_size=hand_size,
        max_lives=3,
    )

    rl_config = lib_rl.RLConfig(
        discount_factor=1.0,
        n_epochs=100_000,
        update_value_model_every_n_episodes=100,
        batch_size=128,
        lr=0.0001,
        replay_buffer_size=1,
    )

    torch.manual_seed(seed)
    game = lib_rl.Game(
        game_config, rl_config, seed=seed, model=lib_agent.SimpleModel(game_config, [])
    )

    state = game.env.reset()

    done = False
    while not done:
        expected_value, expected_index = game.model.forward(
            lib_one_hot.encode(game_config=game_config, game_state=state)
        ).max(axis=-1)

        actual_action, actual_value = game._get_best_action(state)

        assert actual_value == expected_value
        assert (
            index_of_action(actual_action, hand_size=game_config.hand_size)
            == expected_index
        )

        action = game.env.sample_action()
        (state, reward, done) = game.env.step(action)


def test_game_linear_regression_improves_only_affected_action_simple(
    seed=0, n_suits: int = 1, n_ranks: int = 5, hand_size: int = 3
):
    game_config = lib_types.GameConfig(
        n_suits=n_suits,
        n_ranks=n_ranks,
        n_copies=[3] * n_ranks,
        n_players=1,
        n_max_hints=7,
        hand_size=hand_size,
        max_lives=3,
    )

    rl_config = lib_rl.RLConfig(
        discount_factor=1.0,
        n_epochs=100_000,
        update_value_model_every_n_episodes=100,
        batch_size=1,
        lr=0.0001,
        replay_buffer_size=1,
    )

    torch.manual_seed(seed)
    game = lib_rl.Game(
        game_config, rl_config, seed=seed, model=lib_agent.SimpleModel(game_config, [])
    )

    state = game.env.reset()

    done = False
    while not done:
        with torch.no_grad():
            q_values = game.model.forward(
                lib_one_hot.encode(game_config=game_config, game_state=state)
            )
            (points, done, new_state) = game._step(state)

        game.train(n_batches=1)

        with torch.no_grad():
            updated_q_values = game.model.forward(
                lib_one_hot.encode(game_config=game_config, game_state=state)
            )

            print(q_values - updated_q_values)
            assert torch.count_nonzero(q_values - updated_q_values) == 1

            state = new_state


@pytest.mark.skip
@pytest.mark.randomize(
    seed=int, n_suits=int, n_ranks=int, hand_size=int, min_num=1, max_num=100
)
def test_game_linear_regression_improves_only_affected_action_quickcheck(
    seed, n_suits: int, n_ranks: int, hand_size: int
):
    game_config = lib_types.GameConfig(
        n_suits=n_suits,
        n_ranks=n_ranks,
        n_copies=[3] * n_ranks,
        n_players=1,
        n_max_hints=7,
        hand_size=hand_size,
        max_lives=3,
    )

    rl_config = lib_rl.RLConfig(
        discount_factor=1.0,
        n_epochs=100_000,
        update_value_model_every_n_episodes=100,
        batch_size=1,
        lr=0.0001,
        replay_buffer_size=1,
    )

    torch.manual_seed(seed)
    game = lib_rl.Game(
        game_config, rl_config, seed=seed, model=lib_agent.SimpleModel(game_config, [])
    )

    state = game.env.reset()

    done = False
    while not done:
        with torch.no_grad():
            q_values = game.model.forward(
                lib_one_hot.encode(game_config=game_config, game_state=state)
            )
            (points, done, new_state) = game._step(state)

        game.train(n_batches=1)

        with torch.no_grad():
            updated_q_values = game.model.forward(
                lib_one_hot.encode(game_config=game_config, game_state=state)
            )

            print(q_values - updated_q_values)
            assert torch.count_nonzero(q_values - updated_q_values) == 1

            state = new_state
