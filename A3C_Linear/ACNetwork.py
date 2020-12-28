import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim  import Adam


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorCriticNetwork, self).__init__()
        self.FCNet1 = nn.Sequential(nn.Linear(obs_dim, fc1_dim),
                                    nn.ReLU(),
                                    nn.Linear(fc1_dim, fc2_dim),
                                    nn.ReLU()
                                    )
        
        self.FCNet2 = nn.Sequential(nn.Linear(obs_dim, fc1_dim),
                                    nn.ReLU(),
                                    nn.Linear(fc1_dim, fc2_dim),
                                    nn.ReLU()
                                    )
                                    
        self.value_output = nn.Linear(fc2_dim, 1)
        self.probs_output = nn.Sequential(nn.Linear(fc2_dim, action_dim),
                                          nn.Softmax(dim = -1)
                                          )

    def forward(self, obs):
        x1 = self.FCNet1(obs)
        x2 = self.FCNet2(obs)
        probs = self.probs_output(x1)
        value = self.value_output(x2)
        return probs, value