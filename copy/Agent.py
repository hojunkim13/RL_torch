import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Network import DQNNetwork
from Utils.ReplayBuffer import ReplayBuffer
import torch
import numpy as np

class Agent:
    def __init__(self, n_state, n_action, lr, gamma, mem_max, epsilon_decay,epsilon_min, batch_size):
        self.net = DQNNetwork(n_state, n_action)
        self.net_ = DQNNetwork(n_state, n_action)
        self.sync()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)
        self.n_state = n_state
        self.n_action = n_action
        self.actionSpace = [action for action in range(n_action)]
        self.epsilon_min = epsilon_min
        self.epsilon_deacy = epsilon_decay
        self.epsilon = 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_max, n_state, n_action)

    def getAction(self, state, test_mode = False):
        if test_mode or self.epsilon < np.random.rand():
            state = torch.tensor(state, dtype = torch.float32).cuda().unsqueeze(0)
            value = self.net(state)[0]
            action = torch.argmax(value)
            return action.item()
        else:
            action = np.random.choice(self.actionSpace)
            return action
            

    def learn(self):
        if self.memory.mem_cntr <= 300:
            return
        
        S, A, R, S_, D = self.memory.getSample(self.batch_size)
        S = torch.tensor(S, dtype = torch.float).cuda()
        A = torch.tensor(A, dtype = torch.int64).cuda()
        R = torch.tensor(R, dtype = torch.float).cuda()
        S_ = torch.tensor(S_, dtype = torch.float).cuda()
        D = torch.tensor(D, dtype = torch.bool).cuda()
        
        #Bellman Optimization Equation : Q(s, a) <- Reward + max Q(s') * ~done        
        value = torch.gather(self.net(S), dim= 1, index = A)
        target_value = R + self.gamma * torch.max(self.net_(S_), dim = 1)[0].unsqueeze(-1) * ~D
        
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(target_value, value)        
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_deacy
        else:
            self.epsilon = self.epsilon_min        
    
    def sync(self):
        weight_dict = self.net.state_dict()
        self.net_.load_state_dict(weight_dict)

    def load(self, env_name):
        for file_name in os.listdir("./DQN/model"):
            if env_name in file_name:
                weight_dict = torch.load("./DQN/model/" + file_name)
        try:
            self.net.load_state_dict(weight_dict)
            self.net_.load_state_dict(weight_dict)
            print("load success")
        except:
            print("Can't found model weights")
