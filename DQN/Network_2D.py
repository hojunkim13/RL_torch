import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU


class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(DQNNetwork, self).__init__()
        self.n_state = n_state
        self.ConvNet = nn.Sequential(
                                    #96, 96
                                    nn.Conv2d(n_state[0], 64, 4, 2, 1),
                                    nn.ReLU(),
                                    #48, 48
                                    nn.Conv2d(64, 32, 4, 3, 2),
                                    nn.ReLU(),
                                    #17, 17
                                    nn.Conv2d(32, 16, 3, 4, 1),
                                    nn.ReLU(),
                                    #5, 5
                                    nn.Conv2d(16, n_action, 5, 2, 0),
                                    nn.ReLU(),
                                    nn.Flatten(),
        )
        self.cuda()
    
    def forward(self, state):
        values = self.ConvNet(state)
        return values
    

    