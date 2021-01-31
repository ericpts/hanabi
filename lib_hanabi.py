import random
from typing import Tuple, Optional
from lib_types import GameState, GameConfig, ActionType


class HanabiEnvironment(object):
    def __init__(
        self,
        game_config: GameConfig,
        seed: Optional[int] = None,
    ):
        self.game_config = game_config
        if seed is not None:
            random.seed(seed)

    def create_deck(self):
        cards = []
        for i in range(self.game_config.n_suits):
            for j in range(self.game_config.n_ranks):
                for _ in range(self.game_config.n_copies[j]):
                    suit = i
                    number = j
                    card = (suit, number)
                    cards.append(card)
        return cards

    def current_state(self) -> GameState:
        return GameState(
            hands=self.hands,
            stacks=self.stacks,
            remaining_cards=self.remaining_cards,
            lives=self.lives,
            iplayer_to_act=self.iplayer_to_act,
        )

    def reset(self) -> GameState:
        self.deck = self.create_deck()
        random.shuffle(self.deck)

        self.remaining_cards = {}
        for card in self.deck:
            new_value = self.remaining_cards.get(card, 0) + 1
            self.remaining_cards[card] = new_value
        self.hands = []
        for _ in range(self.game_config.n_players):
            self.hands.append(self._draw(self.game_config.hand_size))

        self.stacks = [0 for _ in range(self.game_config.n_suits)]
        self.lives = self.game_config.max_lives
        self.n_hints = self.game_config.n_max_hints

        self.iplayer_to_act = 0
        return self.current_state()

    def _draw(self, n: int):
        ret = []
        for _ in range(n):
            card = self.deck.pop()
            self.remaining_cards[card] -= 1
            ret.append(card)
        return ret

    def _draw_card_for_player(self, player_index):
        if len(self.deck) == 0:
            return
        card = self._draw(1)[0]
        self.hands[player_index].append(card)

    def _apply_action(self, player_index, action):
        reward = -0.1
        (action_type, action_obj) = action
        if action_type == ActionType.PLAY_CARD:
            card_index = action_obj
            card = self.hands[player_index].pop(card_index)
            (suit, number) = card
            expected_number = self.stacks[suit]
            if expected_number == number:
                self.stacks[suit] += 1
                reward = 1.0
            else:
                self.lives -= 1
                reward = -1.0
            self._draw_card_for_player(player_index)
        elif action_type == ActionType.DISCARD:
            card_index = action_obj
            card = self.hands[player_index].pop(card_index)
            self._draw_card_for_player(player_index)
            if self.n_hints < self.game_config.n_max_hints:
                self.n_hints += 1
        else:
            assert action_type == ActionType.HINT
        return reward

    def step(self, action) -> Tuple[GameState, float, bool]:
        reward = self._apply_action(self.iplayer_to_act, action)

        next_player = (self.iplayer_to_act + 1) % self.game_config.n_players

        done = False
        if self.lives == 0:
            done = True
        elif sum(self.stacks) == self.game_config.n_suits * self.game_config.n_ranks:
            done = True
        elif len(self.deck) == 0:
            done = True
        self.iplayer_to_act = next_player
        return (self.current_state(), reward, done)

    def sample_action(self):
        n_cards = len(self.hands[self.iplayer_to_act])
        assert n_cards > 0
        action_type = random.choice(
            [
                ActionType.PLAY_CARD,
                ActionType.DISCARD,
            ]
        )
        action_index = random.randint(0, n_cards - 1)
        return (action_type, action_index)
