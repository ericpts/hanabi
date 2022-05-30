import collections
from enum import Enum


class ActionType(Enum):
    PLAY_CARD = 1
    DISCARD = 2
    HINT = 3

    def __int__(self):
        return self.value

    def short(self):
        if self.value == 1:
            return "P"
        elif self.value == 2:
            return "D"
        assert False, f"Unknown action type: {self}"


GameConfig = collections.namedtuple(
    "GameConfig",
    [
        "n_suits",
        "n_ranks",
        "n_copies",
        "hand_size",
        "n_players",
        "max_lives",
        "n_max_hints",
    ],
)

GameState = collections.namedtuple(
    "GameState",
    ["hands", "stacks", "remaining_cards", "lives", "iplayer_to_act"],
)


def index_of_action(action, hand_size):
    (action_type, card_index) = action
    action_index = card_index
    if action_type == ActionType.DISCARD:
        action_index += hand_size
    return action_index


def action_of_index(index, hand_size):
    if index >= hand_size:
        action_type = ActionType.DISCARD
        index = index - hand_size
    else:
        action_type = ActionType.PLAY_CARD
    card_index = int(index)
    return (action_type, card_index)
