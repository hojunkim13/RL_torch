import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        #96
        self.ConvNet = nn.Sequential(nn.Conv2d(4, 16, 4, 2, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   #48
                                   nn.Conv2d(16, 32, 4, 2, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   #24
                                   nn.Conv2d(32, 64, 4, 2, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   #12
                                   nn.Conv2d(64, 128, 4, 4, 0),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   # 3 * 3
                                   nn.Conv2d(128, 256, 3, 1, 0),
                                   nn.Flatten(),
                                   )

        self.actor = nn.Sequential(nn.Linear(256, action_dim),
                                    nn.Softmax(1))
        
        self.critic = nn.Linear(256, 1)

        self.cuda()

    def forward(self, state):
        feature = self.ConvNet(state)
        policy = self.actor(feature)
        value = self.critic(feature)
        return policy, value

