import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, action_dim),
            nn.Softmax(1)
        )
        self.cuda()

    def forward(self, state):
        policy = self.ConvNet(state)
        return policy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1),
        )
        self.cuda()

    def forward(self, state):
        value = self.ConvNet(state)
        return value
