from Network import Actor, Critic
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Agent:
    def __init__(self, state_dim, action_dim, lr, gamma, lmbda, epsilon,
                 time_step, K_epochs, batch_size):
        self.actor = Actor(state_dim, action_dim)        
        self.critic = Critic(state_dim, action_dim)        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.batch_size = batch_size
        
        self.time_step = time_step
        self.K_epochs = K_epochs
        if type(state_dim) is tuple:
            self.S = np.zeros((time_step, *state_dim), dtype = 'float')
            self.S_ = np.zeros((time_step, *state_dim), dtype = 'float')
        else:
            self.S = np.zeros((time_step, state_dim), dtype = 'float')
            self.S_ = np.zeros((time_step, state_dim), dtype = 'float')
            
        self.A = np.zeros((time_step, 1))
        self.R = np.zeros((time_step, 1), dtype = 'float')
        self.D = np.zeros((time_step, 1), dtype = 'bool')
        self.P = np.zeros((time_step, 1), dtype = 'float')
        self.mntr = 0
        
    
    def get_action(self, state):
        if type(self.state_dim) is tuple:            
            state = torch.Tensor(state).view(-1, *self.state_dim).cuda()
        else:
            state = torch.Tensor(state).view(-1, self.state_dim).cuda()
        with torch.no_grad():
            policy = self.actor(state)
            policy = policy[0].detach().cpu().numpy()
            action = np.random.choice(self.action_space, p = policy)
            prob = policy[action]
        return action, prob

    def store(self, s, a, r, s_, d, a_prob):
        idx = self.mntr
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.P[idx] = a_prob
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
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return advantage, td_target

    def learn(self):
        if self.mntr != self.time_step:
            return
        S = torch.Tensor(self.S[:self.mntr]).cuda()
        A = torch.Tensor(self.A[:self.mntr]).cuda().long()
        R = torch.Tensor(self.R[:self.mntr]).cuda()
        S_= torch.Tensor(self.S_[:self.mntr]).cuda()
        D = torch.Tensor(self.D[:self.mntr]).cuda().bool()
        P = torch.Tensor(self.P[:self.mntr]).cuda()
        
        advantage, td_target = self.get_advantage(S, R, S_, D)

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.time_step)), self.batch_size, False):            
                
                policy = self.actor(S[index])
                td = self.critic(S[index])
                
                prob_new = policy.gather(1, A[index])
                                
                ratio = torch.exp(torch.log(prob_new + 1e-6) - torch.log(P[index] + 1e-6))
                
                surrogate1 = ratio * advantage[index]
                surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage[index]
                
                actor_loss = -torch.min(surrogate1, surrogate2).mean()                
                critic_loss = F.smooth_l1_loss(td, td_target.detach()[index])                
                entropy_loss = torch.distributions.Categorical(policy).entropy().mean()
                total_loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()            
                self.critic_optimizer.step()            
        self.mntr = 0
    
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_a.pt')
        torch.save(self.critic.state_dict(), path + '_c.pt')
        

    def load(self, path):        
        self.actor.load_state_dict(torch.load(path + '_a.pt'))
        self.critic.load_state_dict(torch.load(path + '_c.pt'))