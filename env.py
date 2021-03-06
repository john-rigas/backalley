import gym
from gym import spaces
from gym.utils import seeding
import random
from utils import Deck, CARD_TABLE, RANK_TABLE, SUIT_TABLE
import utils
import numpy as np


RANKINGS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS = ['H','C','D','S']


class BackalleyEnv(gym.Env):
    def __init__(self, num_players = 4, hand = 12):
        self.num_players = num_players
        self.hand = hand
        self.deck = Deck()
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.hand + 1), # bid, 0 - num_players for bid value, irrelevant after bidding
            spaces.MultiDiscrete([len(RANKINGS),len(SUITS)]) # select card, irrelevant during bidding
            ))
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([len(RANKINGS),len(SUITS)]),  # trump
            spaces.Tuple(
                tuple([spaces.Tuple(
                    tuple([spaces.MultiDiscrete([len(RANKINGS)+1,len(SUITS)+1])] * self.num_players)  # cards played, card is 13,4 if not played yet
                )]) * self.hand
            ),
            spaces.Tuple(tuple([spaces.Discrete(self.hand + 2)] * self.num_players)), # bids, 0- hand no for bid vals, hand no + 1 if null
            spaces.Discrete(self.num_players) # seat location
            ))
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.cards_in_hand = [[] for _ in range(self.num_players)]
        self.deck.reset()
        self.round = 0
        self.starting_player = 0
        self._deal()
        self.last_winner = None
        self.cards_played = [['XY' for player in range(self.num_players)] for card in range(self.hand)]
        self.seat_location = random.randint(0, self.num_players - 1) # shouldnt be random
        self.bids = [self.hand + 1 for p in range(0, self.num_players)]
        self._make_random_bids(0, self.seat_location)
        self.trumps_won = [0 for _ in range(self.num_players)]
        self.points = [0 for _ in range(self.num_players)]
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        if self.round == 0:
            if action[0] < 0 or action[0] > self.hand:
                raise utils.BidOutOfBoundsError(self.hand)
            if self.seat_location == self.num_players - 1 and sum(self.bids[:-1]) + action[0] == self.hand:
                raise utils.BidNotAllowedError(action[0], self.hand)
            self.bids[self.seat_location] = action[0]
            self._make_random_bids(self.seat_location + 1, self.num_players)
        else:
            card = self._convert_action_to_card(action[1])
            if card not in self.cards_in_hand[self.seat_location]:
                raise utils.CardNotInHandError
            if self._check_trump(card, self.seat_location):
                raise utils.TrumpNotThrownError(self.trump[1])
            if self._check_leading_suit(card, self.seat_location):
                raise utils.NotLeadingSuitError(self.cards_played[self.round - 1][self.starting_player][1])
            
            self.cards_played[self.round - 1][self.seat_location] = self._throw_card(self.seat_location, card)
            self._finish_round()
            self.last_winner = self._determine_round_winner()
            self.render()
            self.starting_player = self.last_winner
        if self.round == self.hand:
            done = True
            self.points = self._tally_points()
        else:
            self.round += 1
            self._start_new_round()
        return self._get_obs(), self.points[self.seat_location], done, {}

    def render(self):
        print ()
        print ('Card {} of {}'.format(self.round, self.hand))
        print ('Trump: ', self.trump)
        print ()
        for idx in range(self.num_players):
            print ('{}{}{}Player {} threw {}'.format('(S)' if self.starting_player == idx else '   ',
                                                    '(U)' if self.seat_location == idx else '   ',
                                                    '(W) ' if self.last_winner == idx else '    ',
                                                    idx,
                                                    self.cards_played[self.round - 1][idx]))
        print ()

    def _deal(self):
        self.trump = self.deck.draw()
        for _ in range(self.hand):
            for idx in range(self.num_players):
                self.cards_in_hand[idx].append(self.deck.draw())

    def _get_obs(self):
        return (self.trump,
                self._get_discrete_cards_played(),
                self.bids,
                self.seat_location)

    def _get_discrete_cards_played(self):
        return [[self._convert_card(card) for card in r] for r in self.cards_played]

    def _convert_card(self, card):
        return (CARD_TABLE[card[0]], CARD_TABLE[card[1]])

    def _make_random_bids(self, start, end):
        for idx in range(start, end):
            self.bids[idx] = random.randint(0, self.hand//3)

    def _tally_points(self):
        return [t + 10 if t == b else t for t,b in zip(self.trumps_won, self.bids)]

    def _finish_round(self):
        if self.seat_location <self.starting_player:
            players_after = list(range(self.seat_location + 1, self.starting_player))
        else:
            players_after = list(range(self.seat_location + 1, self.num_players)) + list(range(0, self.starting_player))
        for idx in players_after:
            self.cards_played[self.round - 1][idx] = self._throw_card_randomly(idx)

    def _start_new_round(self):
        if self.seat_location < self.starting_player:
            players_before = list(range(self.starting_player, self.num_players)) + list(range(0, self.seat_location))
        else:
            players_before = list(range(self.starting_player, self.seat_location))   
        for idx in players_before:
            self.cards_played[self.round - 1][idx] = self._throw_card_randomly(idx)

    def _determine_round_winner(self):
        order = list(range(self.starting_player, self.num_players)) + list(range(0, self.starting_player))
        cards_thrown = [self.cards_played[self.round - 1][idx] for idx in order]
        winning_player, winning_card = order[0], cards_thrown[0]
        for idx, card in zip(order[1:], cards_thrown[1:]):
            if self._beats(card, winning_card):
                winning_card = card
                winning_player = idx
        self.trumps_won[winning_player] += 1
        return winning_player

    def _throw_card_randomly(self, idx):
        card = random.choice(self.cards_in_hand[idx])
        while self._check_trump(card, idx) or self._check_leading_suit(card, idx):
            card = random.choice(self.cards_in_hand[idx])
        self.cards_in_hand[idx].remove(card)
        return card

    def _throw_card(self, idx, card):
        self.cards_in_hand[idx].remove(card)
        return card

    def _beats(self, new, incumbent):
        if new[1] == self.trump[1] and incumbent[1] != self.trump[1]:
            return True
        if new[1] != incumbent[1]:
            return False
        if CARD_TABLE[new[0]] < CARD_TABLE[incumbent[0]]:
            return False
        return True

    def _convert_action_to_card(self, action):
        return RANK_TABLE[action[0]] + SUIT_TABLE[action[1]]

    @property
    def trump_thrown(self):
        return any(any(c[1] == self.trump[1] for c in _round) for _round in self.cards_played)

    def _check_trump(self, card, idx):
         return (card[1] == self.trump[1] and 
                    not self.trump_thrown and 
                    not all(c[1] == self.trump[1] for c in self.cards_in_hand[idx])
                    and (any(c[1] == self.cards_played[self.round - 1][self.starting_player][1] for c in self.cards_in_hand[idx]) if self.starting_player != idx else True))

    def _check_leading_suit(self, card, idx):
        return ((card[1] != self.cards_played[self.round - 1][self.starting_player][1] and 
                    self.starting_player != idx) and
                    any(c[1] == self.cards_played[self.round - 1][self.starting_player][1] for c in self.cards_in_hand[idx]))

    def _print_user_status(self):
        print ()
        print ('Trump: {}, Your Bid: {}, Hands Won: {} '.format(self.trump,
                                                                self.bids[self.seat_location] if self.round != 0 else None,
                                                                self.trumps_won[self.seat_location]))
        print ()
        for idx in range(self.num_players):
            print ('Player {} with bid of {} threw {}'.format(idx if idx != self.seat_location else 'U',
                                                            self.bids[idx] if self.bids[idx] != self.hand + 1 else '--',
                                                            self.cards_played[self.round - 1][idx] if self.cards_played[self.round - 1][idx] != 'XY' else '--'))     
        print ()                                         
        print ('Your cards are: ')
        cards = self.cards_in_hand[env.seat_location]
        for suit in SUITS:
            print ('{}:  {}'.format(suit, ', '.join(sorted([card for card in cards if card[1] == suit]))))
        print ()
        

if __name__ == '__main__':
    env = BackalleyEnv()
    obs = env.reset()
    for r in range(env.hand + 1):
        env._print_user_status()
        if r == 0:
            no_bid = True
            while no_bid:
                try:
                    obs, reward, done, _ = env.step([int(input("Place bid: ")), np.array([0,0])])
                    no_bid = False
                except utils.BidError:
                    print ('Bidding Error')
        else:
            no_card = True
            while no_card:
                try:
                    card = input("Pick a card to throw: ")
                    if card not in Deck().cards:
                        raise utils.NoCardError
                    discrete_action = [CARD_TABLE[card[0]], CARD_TABLE[card[1]]]
                    obs, reward, done, _ = env.step([0, np.array(discrete_action)])
                    no_card = False
                except utils.CardChoiceError:
                    print ('Card Choice Error')

    print ('Results: ')
    for idx in range(env.num_players):
        print ('Player {}: Bid = {}, Hands Won = {}, Points = {}'.format(idx, env.bids[idx], env.trumps_won[idx], env.points[idx]))
    
