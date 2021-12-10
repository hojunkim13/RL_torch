import os, sys

sys.path.append(".")
from Network import DNN, ConvNet
from Utils.ReplayBuffer import ReplayBuffer
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(
        self, state_dim, action_dim, lr, gamma, mem_max, eps_decay, batch_size
    ):
        if isinstance(state_dim, tuple):
            self.net = ConvNet(state_dim, action_dim).to(device)
            self.target_net = ConvNet(state_dim, action_dim).to(device)
        else:
            self.net = DNN(state_dim, action_dim).to(device)
            self.target_net = DNN(state_dim, action_dim).to(device)

        self.updateTargetNet(tau=1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps_min = 0.05
        self.eps_deacy = eps_decay
        self.eps = 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_max, state_dim, action_dim)

    def getAction(self, state, test=False):
        if test or self.eps < np.random.rand():
            state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            value = self.net(state)[0]
            action = torch.argmax(value)
            return action.item()
        else:
            action = np.random.choice(self.action_dim)
            return action

    def learn(self):
        if self.memory.mem_cntr <= self.memory.mem_max * 0.9:
            return
        S, A, R, S_, D = self.memory.getSample(self.batch_size)
        S = torch.tensor(S, dtype=torch.float).to(device)
        A = torch.tensor(A, dtype=torch.long).to(device)
        R = torch.tensor(R, dtype=torch.float).to(device)
        S_ = torch.tensor(S_, dtype=torch.float).to(device)
        D = torch.tensor(D, dtype=torch.bool).to(device)

        # Bellman Optimization Equation : Q(s, a) <- Reward + max Q(s') * ~done
        value = torch.gather(self.net(S), dim=1, index=A)
        target_value = (
            R + self.gamma * torch.max(self.target_net(S_), dim=1)[0].view(-1, 1) * ~D
        )

        self.optimizer.zero_grad()
        loss = torch.mean(torch.square(value - target_value))
        loss.backward()
        self.optimizer.step()
        self.updateTargetNet()
        if self.eps > self.eps_min:
            self.eps *= self.eps_deacy
        else:
            self.eps = self.eps_min

    def updateTargetNet(self, tau=1e-3):
        mains = self.net.parameters()
        targets = self.target_net.parameters()
        for target_v, main_v in zip(targets, mains):
            target_v.data.copy_(tau * main_v.data + (1.0 - tau) * target_v.data)

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./DQN/weight/{env_name}.pt")

    def load(self, env_name):
        for file_name in os.listdir("./DQN/weight"):
            if env_name in file_name:
                weight_dict = torch.load("./DQN/weight/" + file_name)
        try:
            self.net.load_state_dict(weight_dict)
            self.target_net.load_state_dict(weight_dict)
        except:
            print("Can't found model weights")
