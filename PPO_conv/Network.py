import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.FCNet = nn.Sequential(nn.Linear(state_dim, fc1_dim),
                                   nn.ReLU(),
                                   nn.Linear(fc1_dim, fc2_dim),
                                   nn.ReLU(),
                                   nn.Linear(fc2_dim, action_dim),
                                   nn.Softmax(1)
                                   )
        self.cuda()

    def forward(self, state):
        policy = self.FCNet(state)
        return policy


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.FCNet = nn.Sequential(nn.Linear(state_dim, fc1_dim),
                                   nn.ReLU(),
                                   nn.Linear(fc1_dim, fc2_dim),
                                   nn.ReLU(),
                                   nn.Linear(fc2_dim, 1)
                                   )
        self.cuda()

    def forward(self, state):
        value = self.FCNet(state)
        return value
