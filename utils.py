import random
import numpy as np

RANKINGS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS = ['H','C','D','S']

class Deck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cards = [r + s for r in RANKINGS for s in SUITS]

    def draw(self):
        return self.cards.pop(random.randrange(len(self.cards)))


CARD_TABLE = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
    'H': 0,
    'C': 1,
    'D': 2,
    'S': 3,
    'X': 13,
    'Y': 4
}

RANK_TABLE = {v:k for k,v in CARD_TABLE.items() if k in RANKINGS or k == 'X'}
SUIT_TABLE = {v:k for k,v in CARD_TABLE.items() if k in SUITS or k == 'Y'}

def get_binary(card):
    return CARD_TABLE[card[1]]*13 + CARD_TABLE[card[0]]

def get_binary_from_discrete(card):
    return card[1]*13 + card[0]

def unflatten_obs(obs, num_players, hands):
    intervals = [0, 2, 2+hands*num_players*2, 2+hands*num_players*2+num_players, 
                    2+hands*num_players*2+num_players+1, 2+hands*num_players*2+num_players+1+52, 
                    2+hands*num_players*2+num_players+1+52+4]

    trump = RANK_TABLE[obs[0]] + SUIT_TABLE[obs[1]]
    cards_played = [RANK_TABLE[obs[i]] + SUIT_TABLE[obs[i+1]] for i in range(intervals[1], intervals[2]) if i%2==0]
    bids = obs[intervals[2]:intervals[3]]
    seat = obs[intervals[3]]
    hand = [RANK_TABLE[i%13]+SUIT_TABLE[i//13] for i,c in enumerate(obs[intervals[4]:intervals[5]]) if c == 1]
    throwable = [SUITS[i] if s == 1 else 0 for i,s in enumerate(obs[intervals[5]:])]
    return trump, cards_played, bids, seat, hand, throwable

def flatten_obs(trump, cards_played, bids, seat, hand, throwable):
    trump = np.array(trump)
    cards_played = np.array(cards_played).flatten()
    bids = np.array(bids).flatten()
    seat = np.array(seat).flatten()
    hand = np.array(hand).flatten()
    throwable = np.array(throwable).flatten()
    return np.concatenate([trump, cards_played, bids, seat, hand, throwable])

def unflatten_action(action, bid_options, card_options):
    bid = action // card_options
    card_scalar = action - bid*card_options
    suit = card_scalar // 13
    rank = card_scalar - suit*13
    return [bid, np.array([rank, suit])]

def flatten_action(action, bid_options, card_options):
    bid = action[0]
    suit = action[1][1]
    rank = action[1][0]
    scalar_action = bid*card_options + suit*13 + rank
    return scalar_action

def get_standard_state(hands, num_players):
    return [0]*(2 + num_players*hands*2 + num_players + 1 + 52 + 4)

class BackalleyError(Exception):
    pass

class CardChoiceError(BackalleyError):
    pass

class BidError(BackalleyError):
    pass

class NoCardError(CardChoiceError):
    def __init__(self):
        pass #print ('Card does not exist.  Try another card.')

class CardNotInHandError(CardChoiceError):
    def __init__(self):
        pass #print ('Card is not in your hand.  Try another card.')

class TrumpNotThrownError(CardChoiceError):
    def __init__(self, trump = 'unknown'):
        pass #print ('{} is the trump and has not been thrown.  Try another card.'.format(trump))

class NotLeadingSuitError(CardChoiceError):
    def __init__(self, suit = 'Unknown'):
        pass #print ('The leading suit is {0}.  You must throw a {0} if you have one.'.format(suit))

class BidOutOfBoundsError(BidError):
    def __init__(self, max_bid = 'number of cards'):
        pass #print ('Bid must be between 0 and {}.  Try another bid'.format(max_bid))

class BidNotAllowedError(BidError):
    def __init__(self, bid = '--', bid_max = 'number of cards'):
        pass #print ('You cannot bid {} because bids cannot sum to {}.  Try another bid'.format(bid, bid_max))
