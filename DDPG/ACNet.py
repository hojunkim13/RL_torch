import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, rule):
        super(Actor, self).__init__()
        state_dim = rule.state_dim
        action_dim = rule.action_dim
        self.FCNet = nn.Sequential(nn.Linear(state_dim, rule.fc1_dim),
                                   nn.LayerNorm(rule.fc1_dim),
                                   nn.ReLU(True),
                                   nn.Linear(rule.fc1_dim, rule.fc2_dim),
                                   nn.LayerNorm(rule.fc2_dim),
                                   nn.ReLU(True),
                                   nn.Linear(rule.fc2_dim, action_dim),
                                   nn.Tanh(),
                                   )
        self.cuda()
        
    def forward(self, state):
        return self.FCNet(state)


class Critic(nn.Module):
    def __init__(self, rule):
        super(Critic, self).__init__()
        state_dim = rule.state_dim
        action_dim = rule.action_dim
        self.FCNet = nn.Sequential(nn.Linear(state_dim + action_dim, rule.fc1_dim),
                                   nn.LayerNorm(rule.fc1_dim),
                                   nn.ReLU(True),
                                   nn.Linear(rule.fc1_dim, rule.fc2_dim),
                                   nn.LayerNorm(rule.fc2_dim),
                                   nn.ReLU(True),
                                   nn.Linear(rule.fc2_dim, 1),
                                   )
        self.cuda()
    
    def forward(self, state, action):
        inputs = torch.cat((state,action), dim = -1)
        return self.FCNet(inputs)

def init_weights(params):
    if type(params) == nn.Linear:
        params.weight.data.uniform_(-3e-3, 3e-3)
        params.bias.data.uniform_(-3e-4, 3e-4)