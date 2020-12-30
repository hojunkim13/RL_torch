import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, rule):
        super(Actor, self).__init__()
        # image shape: 3 * 48 * 48
        self.ConvNet = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 24 * 24
                                     nn.Conv2d(32, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 12 * 12
                                     nn.Conv2d(32, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 6 * 6
                                     nn.Flatten(),
                                     nn.Linear(6*6*32, 3),
                                     nn.Tanh(),
                                     )
        self.cuda()

    def forward(self, state):
        return self.ConvNet(state)


class Critic(nn.Module):
    def __init__(self, rule):
        super(Critic, self).__init__()
        self.ConvNet = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 24 * 24
                                     nn.Conv2d(32, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 12 * 12
                                     nn.Conv2d(32, 32, 4, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     # 6 * 6
                                     nn.Flatten(),
                                     nn.Linear(6*6*32, 1),
                                     )

        self.FCNet = nn.Sequential(nn.Linear(3, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1))
        self.cuda()

    def forward(self, state, action):
        state_value = self.ConvNet(state)
        action_value = self.FCNet(action)
        return state_value + action_value
