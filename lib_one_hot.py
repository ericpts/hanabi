import numpy as np
from lib_types import GameState, GameConfig
from lib_util import as_torch
import torch


def _encode_hands(hands, player_index, n_suits: int, n_ranks: int, hand_size: int):
    hands_relative_from_us = hands[player_index:] + hands[:player_index]

    vector_per_player = []
    for player_hand in hands_relative_from_us:
        card_vectors = []
        for (suit, number) in player_hand:
            suit_onehot = np.zeros(n_suits)
            suit_onehot[suit] = 1.0

            number_onehot = np.zeros(n_ranks)
            number_onehot[number] = 1.0

            card_vectors.append(np.concatenate([suit_onehot, number_onehot]))
        for _empty_hand in range(hand_size - len(player_hand)):
            suit_onehot = np.zeros(n_suits)
            number_onehot = np.zeros(n_ranks)
            card_vectors.append(np.concatenate([suit_onehot, number_onehot]))

        vector_per_player.append(card_vectors)
    vector_per_player = np.asarray(vector_per_player)
    return vector_per_player


def _encode_stacks(stacks, n_suits: int, n_ranks: int):
    ret = []
    for istack in range(n_suits):
        stack_onehot = np.zeros(n_ranks + 1)
        stack_onehot[stacks[istack]] = 1.0
        ret.append(stack_onehot)
    ret = np.asarray(ret)
    return ret


def _encode_remaining_cards(remaining_cards, n_suits: int, n_ranks: int):
    ret = np.zeros((n_suits, n_ranks))
    for (suit, number), n_left in remaining_cards.items():
        ret[suit, number] = n_left
    return ret


def _encode_lives(lives: int, max_lives: int):
    ret = np.zeros((max_lives + 1,))
    ret[lives] = 1.0
    return ret


def flatten(*items) -> np.ndarray:
    return np.concatenate([np.asarray(x).flatten() for x in items])


def encode(game_config: GameConfig, game_state: GameState) -> torch.Tensor:
    return as_torch(
        flatten(
            _encode_hands(
                hands=game_state.hands,
                player_index=game_state.iplayer_to_act,
                n_suits=game_config.n_suits,
                n_ranks=game_config.n_ranks,
                hand_size=game_config.hand_size,
            ),
            _encode_stacks(
                stacks=game_state.stacks,
                n_suits=game_config.n_suits,
                n_ranks=game_config.n_ranks,
            ),
            _encode_remaining_cards(
                remaining_cards=game_state.remaining_cards,
                n_suits=game_config.n_suits,
                n_ranks=game_config.n_ranks,
            ),
            _encode_lives(lives=game_state.lives, max_lives=game_config.max_lives),
        )
    )
