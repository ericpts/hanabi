import random
from contextlib import contextmanager
import time
from copy import deepcopy
import numpy as np
import string
from enum import Enum
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy.ma as ma


N_SUITS = 4
N_RANKS = 5
# Add a lot of copies of each card, to make the game very easy.
N_COPIES = np.asarray([3, 2, 2, 2, 1])
N_PLAYERS = 1
N_MAX_HINTS = 7
HAND_SIZE = 5
DISCOUNT_FACTOR = 1.0
N_EPOCHS = 200_000
MAX_LIVES = 3
UPDATE_VALUE_MODEL_EVERY_N_EPISODES = 10
BATCH_SIZE = 64


class EpsilonGreedy(object):
    def __init__(self):
        self.start = 0.9
        self.end = 0.05
        self.decay = 1_000
        self.epoch = 0

    def get(self):
        decay_exponent = self.epoch / self.decay
        eps = self.end + (self.start - self.end) * np.math.exp(-decay_exponent)
        return eps

    def on_epoch_start(self):
        self.epoch += 1


class LogLevel(Enum):
    DEBUG = 1
    ERROR = 2


epsilon_greedy = EpsilonGreedy()
LOG_LEVEL = LogLevel.ERROR


@contextmanager
def timeit(message: str):
    t0 = time.time()
    try:
        yield None
    finally:
        t1 = time.time()
        elapsed = t1 - t0
        print(f"Spent {elapsed * 1_000:.0f}ms in {message}.")


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
    def __init__(self, max_size: int = 100_000):
        self.store = []
        self.at = 0
        self.max_size = max_size

    def push(self, x):
        if len(self.store) == self.max_size:
            self.store[self.at] = x
            self.at += 1
            if self.at == self.max_size:
                self.at = 0
        else:
            self.store.append(x)

    def sample(self, batch_size: int):
        return random.choices(self.store, k=batch_size)

    def __len__(self):
        return len(self.store)


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
        self.pos_number_hints = np.zeros((HAND_SIZE, N_RANKS))
        self.neg_number_hints = np.zeros((HAND_SIZE, N_RANKS))
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

        self.pos_number_hints = add_row(self.pos_number_hints, N_RANKS)
        self.neg_number_hints = add_row(self.neg_number_hints, N_RANKS)

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
            N_PLAYERS * HAND_SIZE * (N_SUITS + N_RANKS)
            # Stacks.
            + N_SUITS * (N_RANKS + 1)
            # Remaining cards in the game.
            + N_SUITS * N_RANKS
            # How many cards the current player has.
            # + (HAND_SIZE + 1)
            # Number of lives left.
            + (MAX_LIVES + 1)
        )
        print(f"Expected input size: {input_size}.")

        action_space = 2 * HAND_SIZE

        self.fc_sizes = [256]

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

    def forward_and_mask(self, inputs):
        X = inputs
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        batch_size = X.shape[0]
        selection_mask = np.tile(np.arange(HAND_SIZE), (batch_size, 1)) < get_hand_size(
            X
        )
        selection_mask = np.concatenate([selection_mask, selection_mask], axis=-1)
        valid_next_actions = ma.masked_where(
            np.logical_not(selection_mask), self.forward(X)
        )
        return valid_next_actions


def encode_state(player_index: int, hands, stacks, remaining_cards, lives: int):
    def encode_hands(hands):
        hands_relative_from_us = hands[player_index:] + hands[:player_index]

        vector_per_player = []
        for player_hand in hands_relative_from_us:
            card_vectors = []
            for (suit, number) in player_hand:
                suit_onehot = np.zeros(N_SUITS)
                suit_onehot[suit] = 1.0

                number_onehot = np.zeros(N_RANKS)
                number_onehot[number] = 1.0

                card_vectors.append(np.concatenate([suit_onehot, number_onehot]))
            for _empty_hand in range(HAND_SIZE - len(player_hand)):
                suit_onehot = np.zeros(N_SUITS)
                number_onehot = np.zeros(N_RANKS)
                card_vectors.append(np.concatenate([suit_onehot, number_onehot]))

            vector_per_player.append(card_vectors)
        vector_per_player = np.asarray(vector_per_player)
        return vector_per_player

    def encode_stacks(stacks):
        ret = []
        for istack in range(N_SUITS):
            stack_onehot = np.zeros(N_RANKS + 1)
            stack_onehot[stacks[istack]] = 1.0
            ret.append(stack_onehot)
        ret = np.asarray(ret)
        return ret

    def encode_remaining_cards(remaining_cards):
        ret = np.zeros((N_SUITS, N_RANKS))
        for (suit, number), n_left in remaining_cards.items():
            ret[suit, number] = n_left
        return ret

    def encode_lives(lives: int):
        ret = np.zeros((MAX_LIVES + 1,))
        ret[lives] = 1.0
        return ret

    def vector_of_state(state):
        return np.asarray(np.concatenate(list(map(lambda arr: arr.flatten(), state))))

    return vector_of_state(
        [
            # Hands must alway scome first, because it's used in finding the hand size
            # for the current player.
            encode_hands(hands),
            encode_stacks(stacks),
            encode_remaining_cards(remaining_cards),
            encode_lives(lives),
        ]
    )


def index_of_action(action):
    (action_type, card_index) = action
    action_index = card_index
    if action_type == ActionType.DISCARD:
        action_index += HAND_SIZE
    return action_index


def action_of_index(index):
    if index >= HAND_SIZE:
        action_type = ActionType.DISCARD
        index -= HAND_SIZE
    else:
        action_type = ActionType.PLAY_CARD
    card_index = int(index)
    return (action_type, card_index)


for action_type in [ActionType.PLAY_CARD, ActionType.DISCARD]:
    for card_index in range(HAND_SIZE):
        action = (action_type, card_index)
        ia = index_of_action(action)
        assert action_of_index(ia) == action


def freeze_model(model):
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False


class NNAgent(BaseAgent):
    def __init__(self, player_index, model):
        super().__init__(player_index)
        self.model = model

    def get_best_action(self, X):
        valid_actions = self.model.forward_and_mask(X)
        best_action = action_of_index(int(valid_actions.argmax(axis=-1)))
        best_score = float(valid_actions.max(axis=-1))
        return (best_action, best_score)

    def on_turn(self, hands, stacks, remaining_cards, lives):
        X_input = encode_state(self.player_index, hands, stacks, remaining_cards, lives)

        if random.random() <= epsilon_greedy.get():
            card_index = random.choice(range(self.hand_size))
            action_type = random.choice([ActionType.PLAY_CARD, ActionType.DISCARD])
            best_action = (action_type, card_index)
            best_score = self.model(X_input)[index_of_action(best_action)]
        else:
            best_action, best_score = self.get_best_action(X_input)
        return X_input, best_action, best_score


def card_to_str(card):
    alphabet = string.ascii_uppercase
    (suit_index, number) = card
    suit = alphabet[suit_index]
    return f"{suit}{number}"


def create_deck():
    cards = []
    for i in range(N_SUITS):
        for j in range(N_RANKS):
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
            self.draw_card_for_player(player_index)
        elif action_type == ActionType.DISCARD:
            card_index = action_obj
            card = self.hands[player_index].pop(card_index)
            self.draw_card_for_player(player_index)
            if self.n_hints < N_MAX_HINTS:
                self.n_hints += 1
        else:
            assert action_type == ActionType.HINT
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
                done = True
                # reward = -10
            else:
                if sum(self.stacks) == N_SUITS * N_RANKS:
                    done = True
                    reward = sum(self.stacks)
                else:
                    done = False
            transition = (X, index_of_action(action), reward, X_new, done)

            log(
                f"P{self.iplayer_to_act}: "
                f"{action_type.short()}"
                f"({card_to_str(card_played)}) "
                f"expected "
                f"{expected_score: .2f}; got {reward: .2f}."
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
        log(f"Obtained total points {total_points}.")
        return transitions


def get_hand_size(X):
    hand_size = np.sum(X[:, : HAND_SIZE * (N_SUITS + N_RANKS)], axis=-1) / 2
    hand_size = hand_size.astype(int)
    hand_size = hand_size[:, np.newaxis]
    return hand_size


def train(value_model, n_batches: int):
    with torch.no_grad():
        samples = replay_buffer.sample(BATCH_SIZE * n_batches)
        (X, action, reward, X_new, done) = map(np.asarray, zip(*samples))

        action = torch.tensor(action)
        value_after_action = value_model.forward_and_mask(X_new).max(axis=-1).data

        add_value_after_action = np.logical_not(done).astype(float)

        one_step_lookahead = torch.tensor(
            reward + add_value_after_action * DISCOUNT_FACTOR * value_after_action
        ).float()
        one_step_lookahead = torch.unsqueeze(one_step_lookahead, -1)

        X = torch.tensor(X).float()

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, action, one_step_lookahead),
        batch_size=BATCH_SIZE,
    )

    losses = []
    for (X, action, one_step_lookahead) in data_loader:
        optimizer.zero_grad()
        predicted = torch.gather(model(X), -1, torch.unsqueeze(action, -1)).float()
        loss = F.smooth_l1_loss(predicted, one_step_lookahead)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return np.mean(losses)


replay_buffer = ReplayBuffer()
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


value_model = deepcopy(model)
for epoch in range(N_EPOCHS):
    epsilon_greedy.on_epoch_start()
    if epoch % 50 == 0:
        LOG_LEVEL = LogLevel.DEBUG
    log(f"Playing one episode at epoch {epoch}: epsilon is {epsilon_greedy.get():.2f}")
    for t in Game(model).play_one_episode():
        replay_buffer.push(t)

    if len(replay_buffer) < 1_000:
        continue

    if epoch % UPDATE_VALUE_MODEL_EVERY_N_EPISODES == 0:
        log("Updating the model for value.")
        value_model = deepcopy(model)
        freeze_model(value_model)

    avg_loss = train(value_model, n_batches=128)
    log(f"Average loss: {avg_loss: .3f}")

    LOG_LEVEL = LogLevel.ERROR
