import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(*args),
            nn.BatchNorm2d(args[1]),
            nn.ReLU(),
        )
        
    def forward(self, x):
        connect = self.block(x)
        output = F.relu(connect + x)
        return output
        


class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(state_dim[0], 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU()
                                        )

        self.res_blocks = nn.Sequential(
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
                            ResidualBlock(256, 256, 1, 1, 0),
        )
                            
    
        self.policy = nn.Sequential(nn.Conv2d(256, 2, 1, 1, 0),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(4*4*2, action_dim),
                                    nn.Softmax(-1),
                                    )

        self.value = nn.Sequential(nn.Conv2d(256, 1, 1, 1, 0),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(4*4*1, 256),
                                    nn.ReLU(),                                    
                                    nn.Linear(256, 1),
                                    nn.Softplus(),
        )

        self.cuda()

    def forward(self, x):
        x = self.conv_block(x)     
        x = self.res_blocks(x)            
        policy = self.policy(x)
        policy = torch.distributions.Categorical(policy)
        value = self.value(x)
        #value = (value + 1.) / 2.
        return policy, value
    