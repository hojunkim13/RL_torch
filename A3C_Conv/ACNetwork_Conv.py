import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim  import Adam


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorCriticNetwork, self).__init__()
                                      # 80 * 80
        self.ConvNet1 = nn.Sequential(nn.Conv2d(4, 32, (7,7), (1,1)),
                                      # 74 * 74
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 64, (6,6), (2,2)),
                                      nn.ReLU(True),
                                      # 35 * 35
                                      nn.Conv2d(64, 128, (5,5), (3,3)),
                                      nn.ReLU(True),
                                      # 11 * 11
                                      nn.Conv2d(128, 64, (5,5), (2,2)),
                                      nn.ReLU(True),
                                      # 4 * 4
                                      nn.Conv2d(64, 1, (4,4), 1),
                                      nn.ReLU(True),
                                      nn.Flatten()
                                      )

                                      # 80 * 80
        self.ConvNet2 = nn.Sequential(nn.Conv2d(4, 32, (7,7), (1,1)),
                                      # 74 * 74
                                      nn.ReLU(True),
                                      nn.Conv2d(32, 64, (6,6), (2,2)),
                                      nn.ReLU(True),
                                      # 35 * 35
                                      nn.Conv2d(64, 128, (5,5), (3,3)),
                                      nn.ReLU(True),
                                      # 11 * 11
                                      nn.Conv2d(128, 64, (5,5), (2,2)),
                                      nn.ReLU(True),
                                      # 4 * 4
                                      nn.Conv2d(64, 1, (4,4), 1),
                                      nn.ReLU(True),
                                      nn.Flatten()
                                      )
        
                                    
        self.value_output = nn.Linear(1, 1)
        self.probs_output = nn.Sequential(nn.Linear(1, action_dim),
                                          nn.Softmax(dim = -1)
                                          )

    def forward(self, obs):
        x1 = self.ConvNet1(obs)
        x2 = self.ConvNet2(obs)
        probs = self.probs_output(x1)
        value = self.value_output(x2)
        return probs, value