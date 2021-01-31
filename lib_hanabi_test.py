from lib_types import ActionType
import lib_hanabi
import pytest
import numpy as np
import random


def test_environment_create_deck():
    game_config = lib_hanabi.GameConfig(
        n_suits=1,
        n_ranks=3,
        n_copies=[2, 1, 1],
        hand_size=1,
        n_players=1,
        max_lives=3,
        n_max_hints=3,
    )
    env = lib_hanabi.HanabiEnvironment(game_config)
    deck = env.create_deck()
    assert deck == [
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 2),
    ]


def test_environment_reset():
    game_config = lib_hanabi.GameConfig(
        n_suits=1,
        n_ranks=3,
        n_copies=[2, 1, 1],
        hand_size=3,
        n_players=1,
        max_lives=3,
        n_max_hints=3,
    )
    env = lib_hanabi.HanabiEnvironment(game_config, seed=0)
    state = env.reset()
    print(state)
    assert state.hands == [
        [
            (0, 2),
            (0, 0),
            (0, 0),
        ]
    ]
    assert state.stacks == [0]
    assert state.lives == 3
    assert state.iplayer_to_act == 0
    assert state.remaining_cards == {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 0,
    }


def test_environment_play():
    game_config = lib_hanabi.GameConfig(
        n_suits=1,
        n_ranks=3,
        n_copies=[2, 1, 1],
        hand_size=3,
        n_players=1,
        max_lives=3,
        n_max_hints=3,
    )
    env = lib_hanabi.HanabiEnvironment(game_config, seed=0)
    env.reset()
    state, reward, done = env.step((ActionType.PLAY_CARD, 1))
    assert reward == 1.0
    assert done == True
    assert state == lib_hanabi.GameState(
        hands=[
            [
                (0, 2),
                (0, 0),
                (0, 1),
            ]
        ],
        stacks=[1],
        remaining_cards={
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
        },
        lives=3,
        iplayer_to_act=0,
    )
