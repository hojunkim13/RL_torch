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
            nn.Flatten(),
        )
        self.DenseNet = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )
        self.ConvNet.apply(self.init_weights)
        self.DenseNet.apply(self.init_weights)
        self.cuda()
    
    def init_weights(self, m):
            if type(m) in (nn.Linear, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    def forward(self, state):
        x = self.ConvNet(state)
        x = self.DenseNet(x)
        return x
    

    