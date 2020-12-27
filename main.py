import random
from copy import deepcopy
import numpy as np
import string
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

SEED = 4
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


N_SUITS = 1
N_CARDS_PER_SUIT = 5
# N_COPIES = [3, 2, 2, 2, 1]
N_COPIES = np.asarray([3, 2, 2, 2, 1]) * 3
N_PLAYERS = 2
N_MAX_HINTS = 7
HAND_SIZE = 2
DISCOUNT_FACTOR = 1.0
EPSILON_GREEDY = 0.1
MAX_LIVES = 3

BATCH_SIZE = 32


class LogLevel(Enum):
    DEBUG = 1
    ERROR = 2


LOG_LEVEL = LogLevel.ERROR


def log(message):
    if LOG_LEVEL == LogLevel.DEBUG:
        print(message)


class Hint(object):
    class HintType(Enum):
        SUIT_HINT = 1
        NUMBER_HINT = 2

    def __init__(
        self,
        hint_type: HintType,
        # Either the suit or the number hinted.
        value: int,
        source_player_index: int,
        target_player_index: int,
    ):
        self.hint_type = hint_type
        self.value = value
        self.source_player_index = source_player_index
        self.target_player_index = target_player_index


class ReplayBuffer(object):
    def __init__(self, size: int = 10_000):
        self.store = [None] * size
        self.size = size
        self.at = 0
        self.filled_once = False

    def push(self, transition):
        self.store[self.at] = transition
        self.at += 1
        if self.at == self.size:
            self.at = 0
            self.filled_once = True

    def sample(self, batch_size: int):
        ret = []
        for _ in range(batch_size):
            if self.filled_once:
                idx = random.randint(0, self.size - 1)
            else:
                idx = random.randint(0, self.at - 1)
            ret.append(self.store[idx])
        return ret

    def __len__(self):
        if self.filled_once == True:
            return self.size
        else:
            return self.at


class ActionType(Enum):
    PLAY_CARD = 1
    DISCARD = 2
    HINT = 3

    def short(self):
        if self.value == 1:
            return "P"
        elif self.value == 2:
            return "D"
        assert False, f"Unknown action type: {self}"


class BaseAgent(object):
    def __init__(self, player_index: int):
        self.player_index = player_index
        self.hand_size = HAND_SIZE
        self.pos_suit_hints = np.zeros((HAND_SIZE, N_SUITS))
        self.neg_suit_hints = np.zeros((HAND_SIZE, N_SUITS))
        self.pos_number_hints = np.zeros((HAND_SIZE, N_CARDS_PER_SUIT))
        self.neg_number_hints = np.zeros((HAND_SIZE, N_CARDS_PER_SUIT))
        pass

    def on_lose_card(self, card_index):
        def remove_card(array):
            return np.delete(array, card_index, 0)

        self.hand_size -= 1
        self.pos_suit_hints = remove_card(self.pos_suit_hints)
        self.neg_suit_hints = remove_card(self.neg_suit_hints)
        self.pos_number_hints = remove_card(self.pos_number_hints)
        self.neg_number_hints = remove_card(self.neg_number_hints)

    def on_discard(self, hands, player_index, card_index):
        if player_index != self.player_index:
            return
        self.on_lose_card(card_index)

    def on_play(self, hands, player_index, card_index):
        if player_index != self.player_index:
            return
        self.on_lose_card(card_index)

    def on_draw_card(self):
        def add_row(array, size: int):
            return np.append(array, [np.zeros((size,))], axis=0)

        self.hand_size += 1

        self.pos_suit_hints = add_row(self.pos_suit_hints, N_SUITS)
        self.neg_suit_hints = add_row(self.neg_suit_hints, N_SUITS)

        self.pos_number_hints = add_row(self.pos_number_hints, N_CARDS_PER_SUIT)
        self.neg_number_hints = add_row(self.neg_number_hints, N_CARDS_PER_SUIT)

    def on_hint(self, hands, hint: Hint):
        if hint.target_player_index != self.player_index:
            return

        if hint.hint_type == Hint.HintType.NUMBER_HINT:
            (pos_array, neg_array) = (self.pos_number_hints, self.neg_number_hints)
        elif hint.hint_type == Hint.HintType.SUIT_HINT:
            (pos_array, neg_array) = (self.pos_suit_hints, self.neg_suit_hints)
        else:
            assert False, f"Unknown hint type: {hint.hint_type}"

        for ic, c in enumerate(hands[self.player_index]):
            (suit, number) = c
            is_in_hint = False
            if hint.hint_type == Hint.HintType.NUMBER_HINT:
                is_in_hint = number == hint.value
            elif hint.hint_type == Hint.HintType.SUIT_HINT:
                is_in_hint = suit == hint.value
            else:
                assert False, f"Unknown hint type: {hint.hint_type}"

            if is_in_hint:
                pos_array[ic][hint.value] = 1.0
            else:
                neg_array[ic][hint.value] = 1.0


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        input_size = (
            # Encoding each player's hand.
            N_PLAYERS * HAND_SIZE * (N_SUITS + N_CARDS_PER_SUIT)
            # Stacks.
            + N_SUITS * (N_CARDS_PER_SUIT + 1)
            # Remaining cards in the game.
            + N_SUITS * N_CARDS_PER_SUIT
            # How many cards the current player has.
            + (HAND_SIZE + 1)
            # Number of lives left.
            + (MAX_LIVES + 1)
        )
        log(f"Expected input size: {input_size}.")

        action_space = 2 * HAND_SIZE

        self.fc_sizes = [500, 500, 500]

        self.fcs = []
        last_size = input_size
        for s in self.fc_sizes:
            self.fcs.append(nn.Linear(last_size, s))
            last_size = s

        self.last_fc = nn.Linear(self.fc_sizes[-1], action_space)

    def forward(self, inputs):
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs).float()
        X = inputs
        for f in self.fcs:
            X = f(X)
            X = F.relu(X)
        X = self.last_fc(X)
        return X


def encode_state(player_index: int, hands, stacks, remaining_cards, lives: int):
    def encode_hands(hands):
        hands_relative_from_us = hands[player_index:] + hands[:player_index]

        vector_per_player = []
        for player_hand in hands_relative_from_us:
            card_vectors = []
            for (suit, number) in player_hand:
                suit_onehot = np.zeros(N_SUITS)
                suit_onehot[suit] = 1.0

                number_onehot = np.zeros(N_CARDS_PER_SUIT)
                number_onehot[number] = 1.0

                card_vectors.append(np.concatenate([suit_onehot, number_onehot]))
            for _empty_hand in range(HAND_SIZE - len(player_hand)):
                suit_onehot = np.zeros(N_SUITS)
                number_onehot = np.zeros(N_CARDS_PER_SUIT)
                card_vectors.append(np.concatenate([suit_onehot, number_onehot]))

            vector_per_player.append(card_vectors)
        vector_per_player = np.asarray(vector_per_player).flatten()
        return vector_per_player

    def encode_stacks(stacks):
        ret = []
        for istack in range(N_SUITS):
            stack_onehot = np.zeros(N_CARDS_PER_SUIT + 1)
            stack_onehot[stacks[istack]] = 1.0
            ret.append(stack_onehot)
        ret = np.asarray(ret).flatten()
        return ret

    def encode_remaining_cards(remaining_cards):
        ret = np.zeros((N_SUITS, N_CARDS_PER_SUIT))
        for (suit, number), n_left in remaining_cards.items():
            ret[suit, number] = n_left
        return ret.flatten()

    def encode_hand_size(hand_size: int):
        ret = np.zeros((HAND_SIZE + 1,))
        ret[hand_size] = 1.0
        return ret

    def encode_lives(lives: int):
        ret = np.zeros((MAX_LIVES + 1,))
        ret[lives] = 1.0
        return ret

    return [
        encode_hands(hands),
        encode_stacks(stacks),
        encode_remaining_cards(remaining_cards),
        encode_hand_size(len(hands[player_index])),
        encode_lives(lives),
    ]


def index_of_action(action):
    (action_type, card_index) = action

    action_index = card_index
    if action_type == ActionType.DISCARD:
        action_index += HAND_SIZE
    return action_index


def get_best_action(X_input, model):
    hand_size = np.argmax(X_input[3])
    y = model(vector_of_state(X_input))
    best_action = None
    best_score: float = 0.0
    for action_type in [ActionType.PLAY_CARD, ActionType.DISCARD]:
        for card_index in range(hand_size):
            ia = index_of_action((action_type, card_index))
            if best_action is None or y[ia] > best_score:
                best_action = (action_type, card_index)
                best_score = y[ia]
    return (best_action, best_score)


def vector_of_state(state):
    return np.concatenate(state)


class NNAgent(BaseAgent):
    def __init__(self, player_index, model):
        super().__init__(player_index)
        self.model = model

    def on_turn(self, hands, stacks, remaining_cards, lives):
        X_input = encode_state(self.player_index, hands, stacks, remaining_cards, lives)
        if random.random() <= EPSILON_GREEDY:
            card_index = random.choice(range(self.hand_size))
            action_type = random.choice([ActionType.PLAY_CARD, ActionType.DISCARD])
            best_action = (action_type, card_index)
            best_score = 0.0
        else:
            best_action, best_score = get_best_action(X_input, self.model)
        return X_input, best_action, best_score


def card_to_str(card):
    alphabet = string.ascii_uppercase
    (suit_index, number) = card
    suit = alphabet[suit_index]
    return f"{suit}{number}"


def create_deck():
    cards = []
    for i in range(N_SUITS):
        for j in range(N_CARDS_PER_SUIT):
            for _ in range(N_COPIES[j]):
                suit = i
                number = j
                card = (suit, number)
                cards.append(card)
    return cards


class Game(object):
    def __init__(self, model):
        self.deck = create_deck()
        random.shuffle(self.deck)

        self.remaining_cards = {}
        for card in self.deck:
            new_value = self.remaining_cards.get(card, 0) + 1
            self.remaining_cards[card] = new_value
        self.hands = []
        for p in range(N_PLAYERS):
            self.hands.append(self.draw(HAND_SIZE))

        self.stacks = [0 for _ in range(N_SUITS)]
        self.lives = MAX_LIVES
        self.n_hints = N_MAX_HINTS

        self.players = []
        for p in range(N_PLAYERS):
            self.players.append(NNAgent(player_index=p, model=model))

        self.iplayer_to_act = 0

    def draw(self, n: int):
        ret = []
        for _ in range(n):
            card = self.deck.pop()
            self.remaining_cards[card] -= 1
            ret.append(card)
        return ret

    def draw_card_for_player(self, player_index):
        if len(self.deck) == 0:
            return
        card = self.draw(1)[0]
        self.hands[player_index].append(card)
        self.players[player_index].on_draw_card()

    def apply_action(self, player_index, action):
        reward = 0.0
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

            self.draw_card_for_player(player_index)

        elif action_type == ActionType.DISCARD:
            card_index = action_obj
            card = self.hands[player_index].pop(card_index)
            self.draw_card_for_player(player_index)

            if self.n_hints < N_MAX_HINTS:
                self.n_hints += 1

            (suit, number) = card
            if self.stacks[suit] > number:
                reward = 0.0
            else:
                if self.remaining_cards[card] == 0:
                    reward = -0.0

        else:
            assert action_type == ActionType.HINT
            reward = 0.0

        return reward

    @torch.no_grad()
    def play_one_episode(self):
        def step():
            X, action, expected_score = self.players[self.iplayer_to_act].on_turn(
                self.hands, self.stacks, self.remaining_cards, self.lives
            )
            (action_type, action_obj) = action

            if action_type == ActionType.PLAY_CARD:
                card_index = action_obj
                for op in self.players:
                    op.on_play(self.hands, self.iplayer_to_act, card_index)
            elif action_type == ActionType.DISCARD:
                card_index = action_obj
                for op in self.players:
                    op.on_discard(self.hands, self.iplayer_to_act, card_index)
            else:
                assert action_type == ActionType.HINT
                hint = action_obj
                for op in self.players:
                    op.on_hint(self.hands, self.iplayer_to_act, hint)

            card_played = self.hands[self.iplayer_to_act][card_index]
            reward = self.apply_action(self.iplayer_to_act, action)

            next_player = (self.iplayer_to_act + 1) % N_PLAYERS
            X_new = encode_state(
                next_player, self.hands, self.stacks, self.remaining_cards, self.lives
            )

            if self.lives == 0:
                log("Ran out of lives.")
                done = True
                reward = -10
            else:
                if sum(self.stacks) == N_SUITS * N_CARDS_PER_SUIT:
                    log("Finished all stacks.")
                    done = True
                    reward = sum(self.stacks)
                else:
                    done = False
            transition = (X, action, reward, X_new, done)

            log(
                f"P{self.iplayer_to_act} chose action "
                f"{action_type.short()}"
                f"({card_to_str(card_played)})"
                f" with expected reward"
                f" {expected_score: .2f}; got {reward: .2f}."
            )
            self.iplayer_to_act = next_player
            return transition

        transitions = []

        done = False
        while len(self.deck) > 0 and self.lives > 0 and not done:
            t = step()
            (X, action, reward, X_new, done) = t
            transitions.append(t)

        if not done:
            assert self.lives > 0
            for _ in range(N_PLAYERS):
                t = step()
                (X, action, reward, X_new, done) = t
                transitions.append(t)
                if done:
                    break

            (X, action, reward, X_new, done) = transitions[-1]
            done = True
            transitions[-1] = (X, action, reward, X_new, done)
            log("Ran out of cards.")

        total_points = sum(self.stacks)
        log(f"obtained total points {total_points}.")
        return transitions


replay_buffer = ReplayBuffer()
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def learn_one_batch(model_for_value):
    @torch.no_grad()
    def construct_batch():
        X_batch = []
        y_batch = []

        for (X, action, reward, X_new, done) in replay_buffer.sample(BATCH_SIZE):
            if not done:
                actual_reward = (
                    reward
                    + DISCOUNT_FACTOR * get_best_action(X_new, model_for_value)[1]
                )
            else:
                actual_reward = reward

            X_vector = vector_of_state(X)

            cy = model_for_value(X_vector)
            action_index = index_of_action(action)
            cy[action_index] = actual_reward
            X_batch.append(torch.from_numpy(X_vector).float())
            y_batch.append(cy)

        X_batch = torch.stack(X_batch)
        y_batch = torch.stack(y_batch)
        assert X_batch.shape[0] == y_batch.shape[0]
        return X_batch, y_batch

    X_batch, y_batch = construct_batch()
    optimizer.zero_grad()
    predicted = model(X_batch)
    loss = criterion(predicted, y_batch)
    loss.backward()
    optimizer.step()
    return float(loss)


for epoch in range(5_000):
    if epoch % 50 == 0:
        LOG_LEVEL = LogLevel.DEBUG
    log("Playing one episode... ")
    transitions = Game(model).play_one_episode()
    for t in transitions:
        replay_buffer.push(t)

    if len(replay_buffer) > 100:
        model_for_value = deepcopy(model)
        losses = []
        for _ in range(32):
            losses.append(learn_one_batch(model_for_value))
        log(f"Average loss: {np.mean(losses): .3f}")

    LOG_LEVEL = LogLevel.ERROR
