import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, action_dim),
        )

    def forward(self, state):
        x = self.net(state)
        return x


class ConvNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ConvNet, self).__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            # 96, 96
            nn.Conv2d(state_dim[0], 64, 4, 2, 1),
            nn.ReLU(),
            # 48, 48
            nn.Conv2d(64, 32, 4, 3, 2),
            nn.ReLU(),
            # 17, 17
            nn.Conv2d(32, 16, 3, 4, 1),
            nn.ReLU(),
            # 5, 5
            nn.Conv2d(16, action_dim, 5, 2, 0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, state):
        values = self.net(state)
        return values
