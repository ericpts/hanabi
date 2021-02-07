import lib_one_hot
import lib_types
import pytest
import numpy as np
import random


def hand_size():
    return list(range(1, 1_000))


@pytest.mark.parametrize("hand_size", [2, 5, 100, 1_000])
def test_action_and_index_conversions(hand_size):
    for action_type in [lib_types.ActionType.PLAY_CARD, lib_types.ActionType.DISCARD]:
        for card_index in range(hand_size):
            action = (action_type, card_index)
            ia = lib_types.index_of_action(action, hand_size)
            assert lib_types.action_of_index(ia, hand_size) == action


def test_encode_lives_simple():
    np.testing.assert_array_equal(
        lib_one_hot._encode_lives(lives=3, max_lives=3), [0, 0, 0, 1.0]
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_lives(lives=2, max_lives=3), [0, 0, 1.0, 0]
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_lives(lives=1, max_lives=3), [0, 1.0, 0, 0]
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_lives(lives=0, max_lives=3), [1.0, 0, 0, 0]
    )


def test_remaining_cards_simple():
    n_suits = 1
    n_ranks = 5
    remaining_cards = {(i, j): 1 for i in range(n_suits) for j in range(n_ranks)}

    np.testing.assert_array_equal(
        lib_one_hot._encode_remaining_cards(remaining_cards, n_suits, n_ranks),
        [[1, 1, 1, 1, 1]],
    )


@pytest.mark.randomize(n_suits=int, n_ranks=int, min_num=1, max_num=100)
def test_remaining_cards_quickcheck(n_suits, n_ranks):
    matrix = np.random.randint(0, 1_000, size=(n_suits, n_ranks))
    encoded = {(i, j): matrix[i, j] for i in range(n_suits) for j in range(n_ranks)}

    np.testing.assert_array_equal(
        lib_one_hot._encode_remaining_cards(encoded, n_suits, n_ranks),
        matrix,
    )


def test_encode_hands_simple():
    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=1,
            n_ranks=3,
            hand_size=2,
            hands=[
                [
                    (0, 0),
                    (0, 1),
                ]
            ],
        ),
        [[[1, 1, 0, 0], [1, 0, 1, 0]]],
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=2,
            n_ranks=3,
            hand_size=2,
            hands=[
                [
                    (0, 0),
                    (1, 1),
                ]
            ],
        ),
        [[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]]],
    )


def test_encode_hands_partially_empty():
    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=1,
            n_ranks=3,
            hand_size=2,
            hands=[
                [
                    (0, 0),
                ]
            ],
        ),
        [[[1, 1, 0, 0], [0, 0, 0, 0]]],
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=2,
            n_ranks=3,
            hand_size=2,
            hands=[
                [
                    (0, 0),
                ]
            ],
        ),
        [[[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
    )

    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=2,
            n_ranks=3,
            hand_size=2,
            hands=[[]],
        ),
        [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
    )


@pytest.mark.randomize(n_suits=int, n_ranks=int, hand_size=int, min_num=1, max_num=100)
def test_encode_hands_quickcheck(n_suits: int, n_ranks: int, hand_size: int):
    hand = [
        (random.randint(0, n_suits - 1), random.randint(0, n_ranks - 1))
        for _ in range(hand_size)
    ]
    encoded_hand = [
        np.concatenate([np.eye(n_suits)[s], np.eye(n_ranks)[r]]) for (s, r) in hand
    ]

    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=n_suits,
            n_ranks=n_ranks,
            hand_size=hand_size,
            hands=[hand],
        ),
        [encoded_hand],
    )


@pytest.mark.randomize(n_suits=int, n_ranks=int, hand_size=int, min_num=1, max_num=100)
def test_encode_hands_empty_hand_quickcheck(n_suits: int, n_ranks: int, hand_size: int):
    hand = []
    encoded_hand = [
        np.concatenate([np.zeros(n_suits), np.zeros(n_ranks)]) for _ in range(hand_size)
    ]

    np.testing.assert_array_equal(
        lib_one_hot._encode_hands(
            player_index=0,
            n_suits=n_suits,
            n_ranks=n_ranks,
            hand_size=hand_size,
            hands=[hand],
        ),
        [encoded_hand],
    )


def test_encode_stacks():
    np.testing.assert_array_equal(
        lib_one_hot._encode_stacks([0], 1, 5), [[1, 0, 0, 0, 0, 0]]
    )

    for at in range(6):
        np.testing.assert_array_equal(
            lib_one_hot._encode_stacks([at], 1, 5), [np.eye(6)[at]]
        )


@pytest.mark.randomize(n_suits=int, n_ranks=int, min_num=1, max_num=100)
def test_encode_stacks_quickcheck(n_suits: int, n_ranks: int):
    stacks = np.random.randint(low=0, high=n_ranks, size=(n_suits,))

    np.testing.assert_array_equal(
        lib_one_hot._encode_stacks(stacks, n_suits, n_ranks),
        np.eye(n_ranks + 1)[stacks],
    )


def test_encode_state_simple():
    game_config = lib_one_hot.GameConfig(
        n_suits=1,
        n_ranks=5,
        n_copies=[3, 2, 2, 2, 1],
        hand_size=1,
        n_players=1,
        max_lives=3,
        n_max_hints=3,
    )
    game_state = lib_types.GameState(
        iplayer_to_act=0,
        hands=[
            [
                (0, 0),
            ]
        ],
        stacks=[0],
        remaining_cards={
            (0, 0): 3,
            (0, 1): 2,
            (0, 2): 2,
            (0, 3): 2,
            (0, 4): 1,
        },
        lives=3,
    )

    np.testing.assert_array_equal(
        lib_one_hot.encode(game_config, game_state),
        lib_one_hot.flatten(
            [[[1, 1, 0, 0, 0, 0]]],
            [[1, 0, 0, 0, 0, 0]],
            [[3, 2, 2, 2, 1]],
            [0, 0, 0, 1.0],
        ),
    )
