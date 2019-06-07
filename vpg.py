from player import Player
import torch
import torch.autograd as autograd
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
from utils import flatten_obs
import utils
import numpy as np
import sys
#import torch.nn.utils as utils
#import torchvision.transforms as T
#from torch.autograd import Variable
from copy import deepcopy

class VPGPlayer(Player):
    def __init__(self, num_inputs, num_outputs, hidden_size = 128, ident = None):
        self.model = PolicyNet(num_inputs, num_outputs, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay= .0005)
        self.ident = ident


    def act(self, obs, playable_cards, round):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        probs = self.model(obs_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()

    def learn(self, obs, actions, rewards):
        new_obs, new_acts, new_rews = [], [], []
        #for o, a, r, i in zip(obs, actions, rewards, range(len(obs))):
            #bid, card = utils.unflatten_action(a, 5, 52)
            #obs_readable = utils.unflatten_obs(o, 4, 4)
            #for ob in obs_readable:
                #print (ob)

            #print ('STATE:', o)
            #print ('ACTION: ', bid, utils.RANK_TABLE[card[0]] + utils.SUIT_TABLE[card[1]])
            #print ('REWARD: ', r)
            #print ()

        self.optimizer.zero_grad()

        state_tensor = torch.tensor(obs, dtype = torch.float, requires_grad = True)
        action_tensor = torch.Tensor(actions).long()
        reward_tensor = torch.Tensor(rewards)

        log_prob = torch.log(self.model(state_tensor))
        selected_log_probs = reward_tensor * log_prob[range(len(action_tensor)), action_tensor]
        loss = -selected_log_probs.mean()
        loss.backward()
        self.optimizer.step() 


class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)


    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        action_scores = self.linear2(x) 
        return F.softmax(action_scores, dim = 1)
