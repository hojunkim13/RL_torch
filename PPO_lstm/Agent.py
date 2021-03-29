from Network import ActorCritic
import numpy as np
import torch
import torch.nn.functional as F

class Agent:
    def __init__(self, state_dim, action_dim, lr, gamma, lmbda, epsilon,
                 buffer_size, batch_size, K_epochs,):
        self.Net = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr = lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.K_epochs = K_epochs

        self.S = np.zeros((buffer_size,)+ state_dim, dtype = 'float')
        self.A = np.zeros((buffer_size, action_dim))
        self.R = np.zeros((buffer_size, 1), dtype = 'float')
        self.S_ = np.zeros((buffer_size,)+ state_dim, dtype = 'float')
        self.D = np.zeros((buffer_size, 1), dtype = 'bool')
        self.P = np.zeros((buffer_size, 1), dtype = 'float')
        self.H_in = np.zeros((buffer_size, 1), dtype = 'float')
        self.H_out = np.zeros((buffer_size, 1), dtype = 'float')
        self.mntr = 0
        
    
    def get_action(self, state):
        state = torch.Tensor(state).cuda()
        with torch.no_grad():
            alpha, beta = self.Net(state)[0]
        distribution = torch.distributions.Beta(alpha, beta)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=1)
        
        action = action.squeeze().cpu().numpy()
        log_prob = log_prob.item()
        return action, log_prob

    def store(self, s, a, a_prob, r, s_, d, h_in, h_out):
        idx = self.mntr

        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.P[idx] = a_prob
        self.H_in[idx] = h_in
        self.H_out[idx] = h_out
        self.mntr += 1

    def get_advantage(self, S, R, S_, D):
        with torch.no_grad():
            _, value = self.Net(S)
            _, value_ = self.Net(S_)
            td_target = R + self.gamma * value_ * ~D
            delta = td_target - value
        advantage = torch.zeros_like(delta)
        running_add = 0
        for i in reversed(range(len(delta))):
            advantage[i] = delta[i] + self.gamma * self.lmbda * running_add
            running_add = advantage[i]
        
        return advantage, td_target

    def learn(self):

        S = torch.Tensor(self.S[:self.mntr]).cuda()
        A = torch.Tensor(self.A[:self.mntr]).cuda()
        R = torch.Tensor(self.R[:self.mntr]).cuda()
        S_= torch.Tensor(self.S_[:self.mntr]).cuda()
        D = torch.Tensor(self.D[:self.mntr]).cuda().bool()
        P = torch.Tensor(self.P[:self.mntr]).cuda()

        
        for k in range(self.K_epochs):
            
            self.optimizer.zero_grad()
            
            advantage, td_target = self.get_advantage(S, R, S_, D)
    
            (alpha,beta), value = self.Net(S)
            dist = torch.distributions.Beta(alpha, beta)
            a_logp = dist.log_prob(A).sum(dim = 1, keepdim = True)
            ratio = torch.exp(a_logp - P)
            
            surrogate1 = ratio * advantage
            surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
            critic_loss = F.smooth_l1_loss(value, td_target.detach())
            
            (actor_loss + critic_loss).backward()
            self.optimizer.step()
        self.mntr = 0

    def save(self, path):
        torch.save(self.Net.state_dict(), path + '.pt')


    def load(self, path):
        self.Net.load_state_dict(torch.load(path + '.pt'))