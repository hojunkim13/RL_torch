import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Network_1D import DQNNetwork
from PER import PrioritizedExperienceReplay
import torch
import numpy as np

class Agent:
    def __init__(self, n_state, n_action, lr, gamma, mem_max, epsilon_decay, batch_size):
        self.net = DQNNetwork(n_state, n_action)
        self.net_ = DQNNetwork(n_state, n_action)
        self.sync()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)
        self.n_state = n_state
        self.n_action = n_action
        self.actionSpace = [action for action in range(n_action)]
        self.epsilon_min = 0.01
        self.epsilon_deacy = epsilon_decay
        self.epsilon = 1
        self.tau = 1e-3
        self.gamma = gamma
        self.batch_size = batch_size
        #self.memory = Memory(mem_max)
        self.memory = PrioritizedExperienceReplay(mem_max)

    def getAction(self, state, test_mode = False):
        if test_mode or self.epsilon < np.random.rand():
            state = torch.tensor(state, dtype = torch.float32).cuda().unsqueeze(0)
            value = self.net(state)[0]
            action = torch.argmax(value)
            return action.item()
        else:
            action = np.random.choice(self.actionSpace)
            return action

    def storeTransition(self, transition):
        state, action, reward, state_, done = transition
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float).unsqueeze(0).cuda()
            state_ = torch.tensor(state_, dtype = torch.float).unsqueeze(0).cuda()
            value = self.net(state)[0][action]
            target_value = reward + self.gamma * torch.max(self.net_(state_), 1)[0].squeeze() * (not done)
            error = torch.abs(target_value - value).cpu().numpy()
        self.memory.add(transition, error)

    def learn(self):
        if self.memory.tree.n_entries < 1000:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_deacy
        else:
            self.epsilon = self.epsilon_min

        data, indice, is_weights = self.memory.sample(self.batch_size)
        data = np.transpose(data)

        S = torch.tensor(np.vstack(data[0]), dtype = torch.float).cuda()
        A = torch.tensor(list(data[1]), dtype = torch.int64).cuda()
        R = torch.tensor(list(data[2]), dtype = torch.float).cuda()
        S_ = torch.tensor(np.vstack(data[3]), dtype = torch.float).cuda()
        D = torch.tensor(list(data[4]), dtype = torch.bool).cuda()
        
        #Bellman Optimization Equation : Q(s, a) <- Reward + max Q(s') * ~done        
        value = torch.gather(self.net(S), dim= 1, index = A.unsqueeze(-1)).squeeze()
        target_value = R + self.gamma * torch.max(self.net_(S_), dim = 1)[0] * ~D
        
        errors = torch.abs(target_value - value).detach().cpu().numpy()
        for idx in range(self.batch_size):
            index = indice[idx]
            self.memory.update(index, errors[idx])
        
        
        self.optimizer.zero_grad()        
        loss = torch.nn.functional.mse_loss(target_value, value)
        loss = (torch.tensor(is_weights).float().cuda() * loss).mean()
        loss.backward()
        self.optimizer.step()        
    

    def sync(self):
        weight_dict = self.net.state_dict()
        self.net_.load_state_dict(weight_dict)

    def softUpdate(self):
        for local_param, target_param in zip(self.net.parameters(), self.net_.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)
            

    def load(self, env_name):
        for file_name in os.listdir("./DQN/model"):
            if env_name in file_name:
                weight_dict = torch.load("./DQN/model/" + file_name)
        try:
            self.net.load_state_dict(weight_dict)
            self.net_.load_state_dict(weight_dict)
        except:
            print("Can't found model weights")
