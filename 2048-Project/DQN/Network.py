import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(DQNNetwork, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(n_state[0], 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, n_action),
        )
        self.cuda()
    
    def forward(self, state):
        x = self.ConvNet(state)
        return x
    

    