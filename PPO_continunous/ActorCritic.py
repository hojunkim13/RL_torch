import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
                                    #4, 96, 96
        self.fcnet = nn.Sequential(nn.Conv2d(4,16,4,2,1),
                                   nn.ReLU(),
                                   # 48 48 
                                   nn.Conv2d(16,32,4,2,1),
                                   nn.ReLU(),
                                   # 24 24
                                   nn.Conv2d(32,64,4,2,1),
                                   nn.ReLU(),
                                   # 12 12
                                   nn.Conv2d(64,128,3,3,0),
                                   nn.ReLU(),
                                   # 4 4
                                   nn.Conv2d(128,256,4,1,0),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   )
        self.fc_alpha = nn.Sequential(nn.Linear(256, action_dim),
                                   nn.Softplus())

        self.fc_beta = nn.Sequential(nn.Linear(256, action_dim),
                                    nn.Softplus())
        self.value = nn.Sequential(nn.Linear(256, 100),
                                   nn.ReLU(),
                                   nn.Linear(100, 1))
        self.apply(self._weights_init)
        self.cuda()

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        x = self.fcnet(state)
        alpha = self.fc_alpha(x) + 1
        beta = self.fc_beta(x) + 1
        value = self.value(x)
        return (alpha, beta), value
    