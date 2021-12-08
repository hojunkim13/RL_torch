import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        )
        self.policy_net = nn.Sequential(nn.Linear(256, action_dim), nn.Softmax(-1))
        self.value_net = nn.Linear(256, 1)

    def forward(self, state):
        x = self.net(state)
        prob = self.policy_net(x)
        value = self.value_net(x)
        return prob, value
