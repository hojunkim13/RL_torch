import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim  import Adam


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.cuda()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        probs = torch.softmax(self.fc3(x), dim= -1)
        return probs

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.cuda()
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values

class ACAgent:
    def __init__(self, obs_dim, action_dim, alpha, beta, gamma):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim, action_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr = alpha)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = beta)
        self.critic_loss_fn = nn.MSELoss()

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).cuda()
        obs = obs.view(-1, self.obs_dim)
        probs = self.actor(obs).squeeze().detach().cpu().numpy()
        action = np.random.choice(self.action_space, p = probs)
        return action

    def train(self, s, a, r, s_, d):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        s = torch.FloatTensor(s).cuda().unsqueeze(0)
        a = int(a)
        action_did = torch.eye(self.action_dim)[a].cuda()
        r = torch.FloatTensor([r]).cuda()
        s_ = torch.FloatTensor(s_).cuda().unsqueeze(0)
        d = torch.FloatTensor([d]).cuda().bool()

        value = self.critic(s)[0]
        value_ = self.critic(s_)[0]
        
        critic_target = r + value_ * self.gamma * ~d
        critic_loss = self.critic_loss_fn(value, critic_target) 
        
        Advantage = critic_target - value
        action_probs = self.actor(s)
        actor_loss = self.actor_loss_fn(action_probs, action_did, Advantage)
        (critic_loss + actor_loss).backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def actor_loss_fn(self, pred, true, Advantage):
        pred = torch.clip(pred, 1e-8, 1-1e-8)
        lik = torch.sum(pred * true)
        log_lik = torch.log(lik)
        return -log_lik * Advantage
        
    def save(self, PATH):
        path_actor = PATH + 'actor.pth'
        path_critic = PATH + 'critic.pth'
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)
    
    def load(self, PATH):
        path_actor = PATH + 'actor.pth'
        path_critic = PATH + 'critic.pth'
        self.actor.load_state_dict(torch.load(path_actor))
        self.critic.load_state_dict(torch.load(path_critic))
