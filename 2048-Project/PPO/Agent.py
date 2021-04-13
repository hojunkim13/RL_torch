import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PPO.Network import Actor, Critic
import numpy as np
import random
import torch
import torch.nn.functional as F
from Environment.DosEnv import _2048
from Environment.Utils import *
from MyMCTS import MCTS


class Agent:
    def __init__(self, state_dim, action_dim, alpha, beta, gamma, lmbda, epsilon,
                 buffer_size, batch_size, k_epochs,):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), beta)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.k_epochs = k_epochs

        self.S = np.zeros((buffer_size, *state_dim), dtype = 'float')
        self.A = np.zeros((buffer_size, 1))
        self.R = np.zeros((buffer_size, 1), dtype = 'float')
        self.S_ = np.zeros((buffer_size, *state_dim), dtype = 'float')
        self.D = np.zeros((buffer_size, 1), dtype = 'bool')
        self.P = np.zeros((buffer_size, 1), dtype = 'float')
        self.mntr = 0

        self.mcts = MCTS(grid = None)
        self.simulator = _2048()
        
    
    def get_action(self, state):
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0).cuda()
        policy = self.actor(state)[0]
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

 
    def store(self, s, a, r, s_, d, log_prob):
        idx = self.mntr
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.P[idx] = log_prob
        self.mntr += 1

    def get_advantage(self, S, R, S_, D):
        with torch.no_grad():
            td_target = R + self.gamma * self.critic(S_) * ~D
            delta = td_target - self.critic(S)
        advantage = torch.zeros_like(delta)
        running_add = 0
        for i in reversed(range(len(delta))):
            advantage[i] = delta[i] + self.gamma * self.lmbda * running_add
            running_add = advantage[i]
        
        return advantage, td_target

    def learn(self):
        if self.mntr != self.buffer_size:
            return

        S = torch.tensor(self.S, dtype = torch.float32).cuda()
        A = torch.tensor(self.A).cuda().long()
        R = torch.tensor(self.R, dtype = torch.float32).cuda()
        S_= torch.tensor(self.S_, dtype = torch.float32).cuda()
        D = torch.tensor(self.D).cuda().bool()
        P = torch.tensor(self.P, dtype = torch.float32).cuda()
        
        advantage, td_target = self.get_advantage(S, R, S_, D)
        indice_pool = list(range(self.buffer_size))
        random.shuffle(indice_pool)
        for k in range(self.k_epochs):
            if len(indice_pool) == 0:
                break
            indice = indice_pool[:self.batch_size]
            indice_pool = indice_pool[self.batch_size:]
    
            policy = self.actor(S[indice])
            prob_new = policy.gather(1, A[indice])
            ratio = torch.exp(torch.log(prob_new) - P[indice])
            
            surrogate1 = ratio * advantage[indice]
            surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage[indice]
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            td = self.critic(S[indice])
            critic_loss = F.smooth_l1_loss(td, td_target[indice].detach())
            entropy_loss = torch.distributions.Categorical(policy).entropy().mean()
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.mntr = 0

    def save(self, env_name):
        path = f"./PPO/data/model/{env_name}"
        torch.save(self.actor.state_dict(), path + '_actor.pt')
        torch.save(self.critic.state_dict(), path + '_critic.pt')

    def load(self, env_name):
        path = f"./PPO/data/model/{env_name}"
        self.actor.load_state_dict(torch.load(path + '_actor.pt'))
        self.critic.load_state_dict(torch.load(path + '_critic.pt'))