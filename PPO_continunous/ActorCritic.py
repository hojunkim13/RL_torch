import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fcnet = nn.Sequential(nn.Linear(state_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU())
        self.fc_mu = nn.Sequential(nn.Linear(256, action_dim),
                                   nn.Tanh())

        self.fc_std = nn.Sequential(nn.Linear(256, action_dim),
                                    nn.Softplus())
        self.value =  nn.Linear(256, 1)
        self.cuda()

    def forward(self, state):
        x = self.fcnet(state)
        mu = 2.0 * self.fc_mu(x)
        std = self.fc_std(x)
        value = self.value(x)
        return (mu, std), value
    