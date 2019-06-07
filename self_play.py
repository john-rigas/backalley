import random
from utils import Deck, CARD_TABLE, RANK_TABLE, SUIT_TABLE, unflatten_action, flatten_obs, flatten_action
import utils
import numpy as np
from rl_env import BackalleyEnv
import torch
from player import Player, RandomPlayer, ManualPlayer
from vpg import VPGPlayer
from copy import deepcopy
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_hand = 4
num_players = 4
env = BackalleyEnv(hand = num_hand, num_players = num_players)
num_episodes = 20000000


batch_obs = {i:[] for i in range(num_players)}
batch_actions = {i:[] for i in range(num_players)}
batch_rewards = {i:[] for i in range(num_players)}
batch_size = 400
batch_counter = 0
BIDS_ACHIEVED = 0
TOTAL_BIDS = 0
mistakes = 0
successes = 0
trys = 0
empties = 0
#maybe figure out a specifc terminal state

players = [VPGPlayer(num_inputs = 2 + env.hand*num_players*2 + num_players + 1 + 52 + 4,
                        num_outputs = (env.hand + 1)*52, ident = i) for i in range(num_players)]

for episode in range(num_episodes):
    #print ('Mistakes: ', mistakes)
    #print ('Successes: ', successes)
    #print ('success rate: ', successes/(mistakes+successes+1))
    #print ('Trys: ', trys)
    #print ('Empties: ', empties)
    #print ('Zeroed: ', trys - empties - mistakes - successes)
    #print ('EPISODE: ', episode)
    try:
        done = False
        obs = [None]*num_players
        new_obs = [None]*num_players
        actions = [None]*num_players
        #mistake = [False]*num_players
        obs[env.starting_player] = flatten_obs(*env.reset())
        round_no = 0
        while not done:
            for i in range(num_players):
                trys += 1
                #env._print_user_status()
                player_no = env.current_seat
                if round_no == 0:
                    #no_bid = True
                    #while no_bid:
                # try:
                    action = players[player_no].act(obs[player_no], None, round_no)
                    actions[player_no] = unflatten_action(action, env.hand + 1, 52)
                    fake_bid = 1 if actions[player_no][0] == 0 else 0
                    next_obs, rewards, done, _ = env.step(actions[player_no])
                    next_obs = flatten_obs(*next_obs)
                        #no_bid = False
                    #except utils.BidError:
                    #   mistake[player_no] = True
                    #  next_obs, rewards, done, _ = env.step([fake_bid, np.array([0,0])])
                    # next_obs = flatten_obs(*next_obs)
                        #no_bid = False
                else:
                    #no_card = True
                    playable_cards = env.get_playable_cards()
                    #while no_card:
                # try:
                    action = players[player_no].act(obs[player_no], playable_cards, round_no)
                    actions[player_no] = unflatten_action(action, env.hand + 1, 52)
                    next_obs, rewards, done, _ = env.step(actions[player_no])
                    next_obs = flatten_obs(*next_obs)
                        #no_card = False
                    #except utils.CardChoiceError:
                    #   mistake[player_no] = True
                    #  next_obs, rewards, done, _ = env.step([0, np.array(playable_cards[0])])
                    # next_obs = flatten_obs(*next_obs)
                        #no_card = False
                next_player = env.current_seat
                #if mistake[next_player]:
                #   mistakes += 1
                #  batch_obs[next_player].append(obs[next_player])
                # batch_actions[next_player].append(flatten_action(actions[next_player], env.hand+1, 52))
                    #batch_rewards[next_player].append(-1000)
                    #batch_counter += 1
                    #mistake[next_player] = False
                #else:
                if obs[next_player] is not None and not done:
                    successes += 1
                    batch_obs[next_player].append(obs[next_player])
                    batch_actions[next_player].append(flatten_action(actions[next_player], env.hand+1, 52))
                    batch_rewards[next_player].append(1)
                    batch_counter += 1
                else:
                    empties += 1
                obs[next_player] = deepcopy(next_obs)
                if batch_counter >= batch_size:
                    for idx in range(num_players):
                        players[idx].learn(batch_obs[idx], batch_actions[idx], batch_rewards[idx])
                    batch_counter = 0
                    batch_obs = {i:[] for i in range(num_players)}
                    batch_actions = {i:[] for i in range(num_players)}
                    batch_rewards = {i:[] for i in range(num_players)}
                    if TOTAL_BIDS > 0:
                        print ('BID PERCENTAGE: ', BIDS_ACHIEVED/TOTAL_BIDS)
            if round_no == env.hand:
                for idx in range(num_players):
                    batch_obs[idx].append(obs[idx])
                    batch_actions[idx].append(flatten_action(actions[idx], env.hand+1, 52))
                    batch_rewards[idx].append(rewards[idx])
                    batch_counter += 1
                    BIDS_ACHIEVED += 1 if rewards[idx] >= 10 else 0
                    TOTAL_BIDS += 1 
            else:
                round_no += 1
    except (utils.CardChoiceError, utils.BidError):
        mistakes += 1
        batch_obs[player_no].append(obs[player_no])
        batch_actions[player_no].append(flatten_action(actions[player_no], env.hand+1, 52))
        batch_rewards[player_no].append(-1)
        batch_counter += 1

    #print ('Results: ')
    #for idx in range(env.num_players):
    #    print ('Player {}: Bid = {}, Hands Won = {}, Points = {}'.format(idx, env.bids[idx], env.trumps_won[idx], env.points[idx]))
