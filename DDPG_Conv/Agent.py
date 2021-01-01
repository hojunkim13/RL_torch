import torch
from torch.optim import Adam
from ACNet import Actor, Critic
from Utils import ReplayBuffer, OUActionNoise
import torch.nn.functional as F
import numpy as np


class Agent:
    def __init__(self, rule, tool):
        self.state_dim = rule.state_dim
        self.action_dim = rule.action_dim

        self.actor = Actor(rule)
        self.actor.apply(tool.init_weights)
        self.actor_target = Actor(rule)
        
        self.critic = Critic(rule)
        self.critic.apply(tool.init_weights)
        self.critic_target = Critic(rule)
        
        self.actor_target.eval()
        self.critic_target.eval()
        self.update_params(1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr = rule.alpha)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = rule.beta)

        self.replaybuffer = ReplayBuffer(rule)
        self.path = './model/' + rule.env_name
        self.gamma = rule.gamma
        self.tau = rule.tau
        self.noise = OUActionNoise(mu = np.zeros(rule.action_dim))
        
        if rule.load == True:
            self.load()


    def get_action(self, state, eval = False):
        action = self.actor(state)[0]
        if eval:
            return action.detach().cpu().numpy()
        noise = torch.Tensor(self.noise()).float().cuda()
        action = torch.clip((action + noise), -1.0, +1.0)
        return action.detach().cpu().numpy()

    def update_params(self, tau = None):
        if tau == None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_target_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        critic_target_state_dict = dict(critic_target_params)
        actor_target_state_dict = dict(actor_target_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                        (1-tau) * critic_target_state_dict[name].clone()
        self.critic_target.load_state_dict(critic_state_dict, strict = False)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                        (1-tau) * actor_target_state_dict[name].clone()
        self.actor_target.load_state_dict(actor_state_dict, strict = False)

    def learn(self):
        if self.replaybuffer.mem_counter < self.replaybuffer.batch_size:
            return
        S, A, R, S_, D = self.replaybuffer.get_samples()
        S  = torch.Tensor(S).cuda()
        A  = torch.Tensor(A).cuda()
        R  = torch.Tensor(R).cuda()
        S_ = torch.Tensor(S_).cuda()
        D  = torch.BoolTensor(D).cuda()

        A_ = self.actor_target(S_)
        values = self.critic(S,A)
        values_ = self.critic_target(S_,A_)
        critic_target = R + self.gamma * values_ * ~D
        
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(values, critic_target)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actions = self.actor(S)
        actions_value = self.critic(S, actions)
        actor_loss = (-1 * actions_value).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.update_params()

    def save(self):
        torch.save(self.actor.state_dict(), self.path + '_actor.pt')
        torch.save(self.critic.state_dict(), self.path + '_critic.pt')
        torch.save(self.actor_target.state_dict(), self.path + '_actor_target.pt')
        torch.save(self.critic_target.state_dict(), self.path + '_critic_target.pt')

    def load(self):
        self.actor.load_state_dict(torch.load(self.path + '_actor.pt'))
        self.critic.load_state_dict(torch.load(self.path + '_critic.pt'))
        self.actor_target.load_state_dict(torch.load(self.path + '_actor_target.pt'))
        self.critic_target.load_state_dict(torch.load(self.path + '_critic_target.pt'))