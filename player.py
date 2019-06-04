import numpy as np
from utils import CARD_TABLE

class Player:
    def __init__(self):
        pass
    
    def act(self, obs, playable_cards):
        pass
        
    def learn(self, obs, action, reward, next_obs):
        print ('ACTION: ', action)
        print ('REWARD: ', reward)

class RandomPlayer(Player):
    def __init__(self):
        pass
    
    def act(self, obs, playable_cards, round):
        trump, cards_played, bids, seat, hand = obs
        if round == 0:
            return [3, np.array([0,0])]
        else:
            action = np.array(playable_cards[0])
            return [0, action]

class ManualPlayer(Player):
    def __init__(self):
        pass

    def act(self, obs, playable_cards, round):
        if round == 0:
            bid = int(input('Place bid: ' ))
            return [bid, np.array([0,0])]
        else:
            card = input('Throw a card: ')
            discrete_action = [CARD_TABLE[card[0]], CARD_TABLE[card[1]]]
            return 0, np.array(discrete_action)
        