import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, rule):
        super(Actor, self).__init__()
        # image shape: 4 * 64 * 64
        self.ConvNet = nn.Sequential(nn.Conv2d(rule.frame_stack, 32, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(32),
                                     # 16 * 16
                                     nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(64),
                                     # 8 * 8
                                     nn.Conv2d(64, 128, 4, 4, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(128),
                                     # 3 * 3
                                     nn.Conv2d(128, 256, 5, 1, 1),
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
        self.ConvNet = nn.Sequential(nn.Conv2d(rule.frame_stack, 32, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(32),
                                     # 16 * 16
                                     nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(64),
                                     # 8 * 8
                                     nn.Conv2d(64, 128, 4, 4, 2),
                                     nn.ReLU(True),
                                     nn.BatchNorm2d(128),
                                     # 3 * 3
                                     nn.Conv2d(128, 256, 5, 1, 1),
                                     # 1 * 1
                                     nn.Flatten(),
                                     )

        self.FCNet = nn.Sequential(nn.Linear(256+3, 30),
                                   nn.ReLU(),
                                   nn.Linear(30, 1))
        self.cuda()

    def forward(self, state, action):
        state_value = self.ConvNet(state)
        inputs = torch.cat([state_value,action], 1)
        return self.FCNet(inputs)
