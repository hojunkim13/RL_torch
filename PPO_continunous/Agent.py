from ActorCritic import ActorCritic
import numpy as np
import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F


class Agent:
    def __init__(self, state_dim, action_dim, lr, epsilon, gamma, lmbda,  timestep, K_epochs):
        self.net = ActorCritic(state_dim, action_dim)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.k_epochs = K_epochs
        self.timestep = timestep
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data = np.empty(timestep, np.dtype([('s', np.float64, (state_dim,)),
                                                 ('a', np.long, (1,)),
                                                 ('log_prob', np.float64, (1,)),
                                                 ('r', np.float64, (1,)),
                                                 ('s_', np.float64, (state_dim,)),
                                                 ('d', np.bool, (1,))
                                                 ]))
        self.mntr = 0                                                

    def get_action(self, state):
        state = torch.Tensor(state).cuda().view(-1,self.state_dim)
        with torch.no_grad():
            policy = self.net(state)[0]
        dist = torch.distributions.categorical.Categorical(policy[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def store(self, transition):
        self.data[self.mntr] = transition
        self.mntr += 1

    def learn(self):
        assert len(self.data) == self.timestep, '타임스텝 t와 데이터의 길이가 다릅니다.'
        
        S = torch.Tensor(self.data['s'].copy()).float().cuda()
        A = torch.Tensor(self.data['a'].copy()).long().cuda()
        log_prob_old = torch.Tensor(self.data['log_prob'].copy()).float().cuda()
        R = torch.Tensor(self.data['r'].copy()).float().cuda()
        S_ = torch.Tensor(self.data['s_'].copy()).float().cuda()
        D = torch.Tensor(self.data['d'].copy()).bool().cuda()

        for i in range(self.k_epochs):
            self.optimizer.zero_grad()
            td_target, advantage = self.get_advantage(S,R,S_,D)
            policy, value = self.net(S)
            prob_new = policy.gather(1, A)
            log_prob_new = torch.log(prob_new)
            ratio = torch.exp(log_prob_new - log_prob_old)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
            a_loss = -torch.min(surrogate1, surrogate2).mean()
            v_loss = F.smooth_l1_loss(value, td_target.detach())
            (a_loss + v_loss).backward()
            self.optimizer.step()
        self.mntr = 0 
        

    def get_advantage(self, S, R, S_, D):
        with torch.no_grad():
            td_target = R + self.gamma * self.net(S_)[1] * ~D
            delta = td_target - self.net(S)[1]
        advantage = torch.zeros_like(delta)
        running_add = 0
        for i in reversed(range(len(delta))):
            advantage[i] = delta[i] + running_add * self.gamma * self.lmbda
            running_add = advantage[i]
        return td_target, advantage
