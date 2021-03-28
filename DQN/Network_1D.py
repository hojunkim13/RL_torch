import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.fc2 = nn.Linear(32, n_action)        
        self.cuda()
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

    