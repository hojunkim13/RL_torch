import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        if type(state_dim) is tuple:
                                    #80, 80
            self.net = nn.Sequential(nn.Conv2d(state_dim[0],16,4,4,0),
                                   nn.ReLU(),
                                   # 20 20 
                                   nn.Conv2d(16,32,4,2,0),
                                   nn.ReLU(),
                                   # 9 9
                                   nn.Conv2d(32,64,3,2,1),
                                   nn.ReLU(),
                                   # 5 5
                                   nn.Conv2d(64,64,5,1,0),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   )
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, 64,),
                                    nn.ReLU(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    )

        self.policy = nn.Sequential(nn.Linear(64, action_dim),
                                   nn.Softmax(1))

        self.apply(self._weights_init)
        self.cuda()

    @staticmethod
    def _weights_init(m):
        if type(m) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        x = self.net(state)
        policy = self.policy(x)
        return policy
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        if type(state_dim) is tuple:
                                    #80, 80
            self.net = nn.Sequential(nn.Conv2d(state_dim[0],16,4,4,0),
                                   nn.ReLU(),
                                   # 20 20 
                                   nn.Conv2d(16,32,4,2,0),
                                   nn.ReLU(),
                                   # 9 9
                                   nn.Conv2d(32,64,3,2,1),
                                   nn.ReLU(),
                                   # 5 5
                                   nn.Conv2d(64,64,5,1,0),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   )
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, 64,),
                                    nn.ReLU(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    )

        self.value = nn.Sequential(nn.Linear(64, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
        self.apply(self._weights_init)
        self.cuda()

    @staticmethod
    def _weights_init(m):
        if type(m) in (nn.Conv2d, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        x = self.net(state)
        value = self.value(x)
        return value
        