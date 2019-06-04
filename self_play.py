import random
from utils import Deck, CARD_TABLE, RANK_TABLE, SUIT_TABLE
import utils
import numpy as np
from rl_env import BackalleyEnv
import torch
from player import Player, RandomPlayer, ManualPlayer
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BackalleyEnv(hand = 4)
num_episodes = 2000
num_players = 4
#maybe figure out a specifc terminal state

players = [RandomPlayer()]*(num_players-1) + [ManualPlayer()]

for episode in range(num_episodes):
    print ('EPISODE: ', episode)
    done = False
    obs = [None]*num_players
    new_obs = [None]*num_players
    actions = [None]*num_players
    mistake = None
    obs[env.starting_player] = env.reset()
    round_no = 0
    while not done and mistake is None:
        for i in range(num_players):
            env._print_user_status()
            player_no = env.current_seat
            if round_no == 0:
                no_bid = True
                while no_bid:
                    try:
                        actions[player_no] = players[player_no].act(obs[player_no], None, round_no)
                        next_obs, rewards, done, _ = env.step(actions[player_no])
                        no_bid = False
                    except utils.BidError:
                        print ('Bidding Error')
                        mistake = player_no
                        next_obs, rewards, done, _ = env.step([env.hand, np.array([0,0])])
                        no_bid = False
            else:
                no_card = True
                playable_cards = env.get_playable_cards()
                while no_card:
                    try:
                        actions[player_no] = players[player_no].act(obs[player_no], playable_cards, round_no)
                        next_obs, rewards, done, _ = env.step(actions[player_no])
                        #new_obs[(player_no + 1)%num_players]
                        no_card = False
                    except utils.CardChoiceError:
                        print ('Card Choice Error')
                        mistake = player_no
                        next_obs, rewards, done, _ = env.step([0, np.array(playable_cards[0])])
                        no_card = False
            next_player = env.current_seat
            if mistake is not None:
                players[mistake].learn(obs[mistake], actions[mistake], -1000, next_obs)
            else:
                if obs[player_no] and not done:
                    players[next_player].learn(obs[next_player], actions[next_player], 0, next_obs)
            obs[next_player] = deepcopy(next_obs)
            #player_no = (player_no + 1) % num_players
        if round_no == env.hand:
            for idx in range(num_players):
                players[idx].learn(obs[idx], actions[idx], rewards[idx], next_obs) # change terminal state
        else:
            round_no += 1

    print ('Results: ')
    for idx in range(env.num_players):
        print ('Player {}: Bid = {}, Hands Won = {}, Points = {}'.format(idx, env.bids[idx], env.trumps_won[idx], env.points[idx]))

    #[int(input("Place bid: ")), np.array([0,0])]
    #input("Pick a card to throw: ")
    #discrete_action = [CARD_TABLE[card[0]], CARD_TABLE[card[1]]]
    #[0, np.array(discrete_action)]