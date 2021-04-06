import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(DQNNetwork, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(n_state[0], 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, n_action, 4, 1, 0),
            nn.Flatten(),
        )
        self.ConvNet.apply(init_weights)
        self.cuda()
    
    def forward(self, state):
        x = self.ConvNet(state)
        return x
    
    
def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)