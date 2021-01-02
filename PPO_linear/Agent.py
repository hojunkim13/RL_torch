from Network import Actor, Critic
import numpy as np
import torch
import torch.nn.functional as F

class Agent:
    def __init__(self, state_dim, action_dim, alpha, beta, gamma, lmbda, epsilon,
                 time_step, K_epochs,):
        self.actor = Actor(state_dim, action_dim, 128, 128)
        self.critic = Critic(state_dim, action_dim, 128, 128)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), beta)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.time_step = time_step
        self.K_epochs = K_epochs

        self.S = np.zeros((time_step, state_dim), dtype = 'float')
        self.A = np.zeros((time_step, 1))
        self.R = np.zeros((time_step, 1), dtype = 'float')
        self.S_ = np.zeros((time_step, state_dim), dtype = 'float')
        self.D = np.zeros((time_step, 1), dtype = 'bool')
        self.P = np.zeros((time_step, 1), dtype = 'float')
        self.mntr = 0
        
    
    def get_action(self, state):
        state = torch.Tensor(state).view(-1, self.state_dim).cuda()
        policy = self.actor(state)[0].detach().cpu().numpy()
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

    def get_advantage(self, S, A, R, S_, D):
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

        S = torch.Tensor(self.S[:self.mntr]).cuda()
        A = torch.Tensor(self.A[:self.mntr]).cuda().long()
        R = torch.Tensor(self.R[:self.mntr]).cuda()
        S_= torch.Tensor(self.S_[:self.mntr]).cuda()
        D = torch.Tensor(self.D[:self.mntr]).cuda().bool()
        P = torch.Tensor(self.P[:self.mntr]).cuda()

        
        for k in range(self.K_epochs):
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            advantage, td_target = self.get_advantage(S, A, R, S_, D)
    
            policy = self.actor(S)
            prob_new = policy.gather(1, A)
            ratio = torch.exp(torch.log(prob_new) - torch.log(P))
            
            surrogate1 = ratio * advantage
            surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            td = self.critic(S)
            critic_loss = F.smooth_l1_loss(td, td_target.detach())
            
            (actor_loss + critic_loss).backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.mntr = 0

    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pt')
        torch.save(self.critic.state_dict(), path + '_critic.pt')


    def load(self, path):
        self.actor.load_state_dict(torch.load(path + '_actor.pt'))
        self.critic.load_state_dict(torch.load(path + '_critic.pt'))