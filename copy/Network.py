import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(DQNNetwork, self).__init__()
        self.fcNet = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_action),        
        )
        self.cuda()
    
    def forward(self, state):
        x = self.fcNet(state)
        return x
    

    