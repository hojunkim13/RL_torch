import os, sys
path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(path.replace("\\2048-Project",""))
from Network import DQNNetwork
from Utils.ReplayBuffer import ReplayBuffer
import torch
import numpy as np


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, mem_max,
    epsilon_decay, epsilon_min, batch_size, learning_step, tau):
        self.net = DQNNetwork(n_state, n_action)
        self.net_ = DQNNetwork(n_state, n_action)
        self.net_.eval()
        self.update()
        self.tau = tau
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
        self.learning_step = learning_step
        self.step = 0

    def getAction(self, state, test_mode = False):
        if test_mode or self.epsilon < np.random.rand():
            with torch.no_grad():
                state = torch.tensor(state, dtype = torch.float32).cuda().unsqueeze(0)
                value = self.net(state)[0]
            action = torch.argmax(value)
            return action.item()
        else:
            action = np.random.choice(self.actionSpace)
            return action
            

    def learn(self):
        if self.memory.mem_cntr <= self.batch_size:
            self.step = 0 
            return
            
        if self.step != self.learning_step:
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
        loss = torch.nn.functional.smooth_l1_loss(target_value, value)
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.softUpdate()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_deacy
        else:
            self.epsilon = self.epsilon_min
        self.step = 0
    
    def softUpdate(self):
        for target_param, local_param in zip(self.net_.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)
    
    def update(self):
        weight_dict = self.net.state_dict()
        self.net_.load_state_dict(weight_dict)

    def load(self, env_name):
        try:
            for file_name in os.listdir("./2048-Project/model"):
                if env_name in file_name and "DQN" in file_name:
                    weight_dict = torch.load("./2048-Project/model/" + file_name)
                    break
            self.net.load_state_dict(weight_dict)
            self.net_.load_state_dict(weight_dict)
            print(f"load success, filename : {file_name}")
        except:
            print("Can't found model weights")

    def save(self, env_name):
        os.makedirs("./2048-Project/model", exist_ok=True)
        file_name = f"./2048-Project/model/{env_name}_DQN.pt"
        torch.save(self.net_.state_dict(), file_name)
        
