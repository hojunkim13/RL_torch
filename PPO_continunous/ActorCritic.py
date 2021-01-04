import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fcnet = nn.Sequential(nn.Linear(state_dim, 256),
                                   nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(256, action_dim),
                                    nn.Softmax(dim = 1))
        self.value =  nn.Linear(256, 1)
        self.cuda()

    def forward(self, state):
        x = self.fcnet(state)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value
    