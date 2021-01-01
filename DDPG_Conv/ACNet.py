import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, rule):
        super(Actor, self).__init__()
        # image shape: 4 * 96 * 96
        self.ConvNet = nn.Sequential(nn.Conv2d(rule.frame_stack, 16, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(16),
                                     # 48 * 48
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(32),
                                     # 24 * 24
                                     nn.Conv2d(32, 64, 4, 4, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(64),
                                     # 7 * 7
                                     nn.Conv2d(64, 128, 5, 3, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(128),
                                     # 3 * 3
                                     nn.Conv2d(128, 256, 3, 1, 0),
                                     # 1 * 1
                                     nn.Flatten(),
                                     nn.Linear(1*1*256, rule.action_dim),
                                     )
        self.cuda()

    def forward(self, state):
        action = self.ConvNet(state)
        action[:,0] = torch.tanh(action[:,0])
        action[:,1] = torch.sigmoid(action[:,1])
        action[:,2] = torch.sigmoid(action[:,2])
        return action


class Critic(nn.Module):
    def __init__(self, rule):
        super(Critic, self).__init__()
        self.ConvNet = nn.Sequential(nn.Conv2d(rule.frame_stack, 16, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(16),
                                     # 48 * 48
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(32),
                                     # 24 * 24
                                     nn.Conv2d(32, 64, 4, 4, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(64),
                                     # 7 * 7
                                     nn.Conv2d(64, 128, 5, 3, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(128),
                                     # 2 * 2
                                     nn.Conv2d(128, 256, 3, 1, 0),
                                     # 1 * 1
                                     nn.Flatten(),
                                     )

        self.latentNet = nn.Sequential(nn.Linear(3,128),
                                       nn.ReLU(),
                                       nn.Linear(128, 12),
                                       )

        self.FCNet = nn.Sequential(nn.Linear(256+12, 60),
                                   nn.ReLU(),
                                   nn.Linear(60, 1))
        self.cuda()

    def forward(self, state, action):
        state_value = self.ConvNet(state)
        latent = self.latentNet(action)
        inputs = torch.cat([state_value,latent], 1)
        return self.FCNet(inputs)
