import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from DQN.Network import DQNNetwork
from DQN.PER import PrioritizedExperienceReplay
import torch
import numpy as np


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, mem_max,
    epsilon_decay, epsilon_min, decay_step, batch_size, tau):
        self.net = DQNNetwork(n_state, n_action)
        self.net_ = DQNNetwork(n_state, n_action)
        self.net_.eval()
        self.update()
        self.tau = tau
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)
        self.n_state = n_state
        self.n_action = n_action
        self.actionSpace = [action for action in range(n_action)]
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_step = decay_step
        self.decay_cntr = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = PrioritizedExperienceReplay(mem_max)


    def getAction(self, state, test_mode = False):
        if test_mode or self.epsilon < np.random.rand():
            with torch.no_grad():
                state = torch.tensor(state, dtype = torch.float32).cuda().unsqueeze(0)
                value = self.net(state)[0]
            action = torch.argmax(value).item()
            value = value.cpu().numpy()[action]
            return action, value
        else:
            action = np.random.choice(self.actionSpace)
            with torch.no_grad():
                state = torch.tensor(state, dtype = torch.float32).cuda().unsqueeze(0)
                value = self.net(state)[0]
                value = value.cpu().numpy()[action]
            return action, value
            
    def storeTransition(self, *transition):
        s, a, r, s_, d = transition
        s = torch.tensor(s).cuda().unsqueeze(0).float()
        s_ = torch.tensor(s_).cuda().unsqueeze(0).float()
        a = torch.tensor(a).cuda()
        with torch.no_grad():
            pred_value = self.net(s)[0][a]
            target_value = r + self.gamma * torch.max(self.net_(s_),dim =1)[0] * (not d)
            error = np.abs((target_value - pred_value).cpu().numpy())

        self.memory.add(transition, error)


    def adjustHyperparam(self):
        self.decay_cntr += 1        
        if self.decay_cntr % self.decay_step != 0:
            return
        self.decay_cntr = 0
        #epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min        
        
    def learn(self):
        if self.memory.tree.n_entries < 2500:
            return
        
        self.adjustHyperparam()

        data, indice, is_weight = self.memory.sample(self.batch_size)
        data = np.transpose(data)

        S = torch.tensor(np.vstack(data[0]), dtype = torch.float).cuda().view(-1,*self.n_state)
        A = torch.tensor(list(data[1]), dtype = torch.int64).cuda()
        R = torch.tensor(list(data[2]), dtype = torch.float).cuda()
        S_ = torch.tensor(np.vstack(data[3]), dtype = torch.float).cuda().view(-1,*self.n_state)
        D = torch.tensor(list(data[4]), dtype = torch.bool).cuda()
        
        #Bellman Optimization Equation : Q(s, a) <- Reward + max Q(s') * ~done        
        value = torch.gather(self.net(S), dim= 1, index = A.unsqueeze(-1)).squeeze()
        target_value = R + self.gamma * torch.max(self.net_(S_), dim = 1)[0]* ~D
        
        
        errors = target_value - value
        errors = errors.detach().cpu().numpy()
        
        for index, error in zip(indice, errors):
            self.memory.update(index, error)

        loss = torch.nn.functional.smooth_l1_loss(target_value, value)
        total_loss = (torch.tensor(is_weight).cuda() * loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    
    def softUpdate(self):
        for target_param, local_param in zip(self.net_.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)
    
    def update(self):
        weight_dict = self.net.state_dict()
        self.net_.load_state_dict(weight_dict)

    def load(self, env_name, hint = ""):
        try:
            for file_name in os.listdir("./data/model/"):
                if env_name in file_name and "DQN" in file_name:
                    if hint == "":
                        weight_dict = torch.load("./data/model/" + file_name)
                    else:
                        if hint in file_name:
                            weight_dict = torch.load("./data/model/" + file_name)
        
            self.net.load_state_dict(weight_dict)
            self.net_.load_state_dict(weight_dict)
            self.epsilon = self.epsilon_min
            print(f"load success, filename : {file_name}")
        except:
            print("Can't found model weights")

    def save(self, env_name):
        os.makedirs("./data/model/", exist_ok=True)
        file_name = f"./data/model/{env_name}_DQN.pt"
        torch.save(self.net_.state_dict(), file_name)
        
