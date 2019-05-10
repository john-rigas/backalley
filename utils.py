import random

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

RANK_TABLE = {v:k for k,v in CARD_TABLE.items() if k in RANKINGS}
SUIT_TABLE = {v:k for k,v in CARD_TABLE.items() if k in SUITS}

class BackalleyError(Exception):
    pass

class CardChoiceError(BackalleyError):
    pass

class BidError(BackalleyError):
    pass

class NoCardError(CardChoiceError):
    def __init__(self):
        print ('Card does not exist.  Try another card.')

class CardNotInHandError(CardChoiceError):
    def __init__(self):
        print ('Card is not in your hand.  Try another card.')

class TrumpNotThrownError(CardChoiceError):
    def __init__(self, trump = 'unknown'):
        print ('{} is the trump and has not been thrown.  Try another card.'.format(trump))

class NotLeadingSuitError(CardChoiceError):
    def __init__(self, suit = 'Unknown'):
        print ('The leading suit is {0}.  You must throw a {0} if you have one.'.format(suit))

class BidOutOfBoundsError(BidError):
    def __init__(self, max_bid = 'number of cards'):
        print ('Bid must be between 0 and {}.  Try another bid'.format(max_bid))

class BidNotAllowedError(BidError):
    def __init__(self, bid = '--', bid_max = 'number of cards'):
        print ('You cannot bid {} because bids cannot sum to {}.  Try another bid'.format(bid, bid_max))
