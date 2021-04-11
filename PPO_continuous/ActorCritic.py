import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, std = 0.0):
        super(ActorCritic, self).__init__()
                                    #80 80
        self.fcnet = nn.Sequential(nn.Conv2d(state_dim[0],32,4,4,0),
                                   nn.ReLU(),
                                   # 20 
                                   nn.Conv2d(32,64,4,4,2),
                                   nn.ReLU(),
                                   # 6
                                   nn.Conv2d(64,128,4,2,2),
                                   nn.ReLU(),                                
                                   # 4 4
                                   nn.Conv2d(128,256,4,1,0),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   )
        self.fc_mean = nn.Linear(256, action_dim)
                                   

        self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
        self.value = nn.Sequential(nn.Linear(256, 100),
                                   nn.ReLU(),
                                   nn.Linear(100, 1))
        self.apply(self._weights_init)
        self.cuda()

    @staticmethod
    def _weights_init(m):
        if type(m) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        

    def forward(self, state):
        x = self.fcnet(state)
        value = self.value(x)
        mean = self.fc_mean(x)
        std = self.log_std.exp().expand_as(mean)
        return (mean, std), value
    