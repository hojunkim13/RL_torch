import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        #96
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 64),
                                 nn.ReLU()
        )
        self.lstm = nn.LSTM(64, 32)
                                

        self.fc = nn.Sequential(nn.Linear(256, 100),
                                   nn.ReLU())

        self.probs = nn.Sequential(nn.Linear(32, action_dim),
                                   nn.Softmax(dim = -1))
                                       
        
        self.critic = nn.Linear(32, 1)
                                    
        self.cuda()

    def forward(self, state, hidden):
        x = self.fc1(state)
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        value = self.critic(x)
        probs = self.probs(x)
        
        return probs, value, lstm_hidden

        

