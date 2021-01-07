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
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   )

        self.fc = nn.Sequential(nn.Linear(256, 100),
                                   nn.ReLU())

        self.alpha = nn.Sequential(nn.Linear(100,3),
                                   nn.Softplus())

        self.beta = nn.Sequential(nn.Linear(100,3),
                                   nn.Softplus())                                   

        self.critic = nn.Sequential(nn.Linear(256, 100),
                                    nn.ReLU(),
                                    nn.Linear(100,1))

        self.cuda()

    def forward(self, state):
        feature = self.ConvNet(state)
        value = self.critic(feature)

        x = self.fc(feature)
        alpha = self.alpha(x) + 1
        beta = self.beta(x) + 1
        return (alpha, beta), value

