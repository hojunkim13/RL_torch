import numpy as np
import torch
import torch.nn as nn
from torch.optim  import Adam
from ACNetwork_Conv import ActorCriticNetwork
import torch.multiprocessing as mp
import gym
from collections import deque

class ACAgent(mp.Process):
    def __init__(self,globalnet, optimizer, id, Rule):
        super(ACAgent, self).__init__()
        self.rule = Rule
        self.beta = Rule.beta
        self.id = id
        self.action_space = Rule.action_space
        self.obs_tmp = deque(maxlen=4)

        self.globalnet = globalnet
        self.optimizer = optimizer
        self.localnet = ActorCriticNetwork(Rule.obs_dim, Rule.action_dim, Rule.fc1_dim, Rule.fc2_dim)
        self.sync()
        self.gamma = Rule.gamma
        self.G_episode, self.G_episode_score, self.score_queue = Rule.G_episode, Rule.G_episode_score, Rule.score_queue
        self.name = f'Agent[{id:02d}]'
        
        self.env = gym.make(Rule.ENV_NAME)
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.critic_loss_fn = nn.MSELoss()
        

    def actor_loss_fn(self, pred, true, Advantage):
        pred = torch.clip(pred, 1e-8, 1-1e-8)  
        
        lik1 = torch.sum(true * torch.log(pred), dim = -1)
        Advantage_loss = (-lik1 * Advantage).mean()
        lik2 = pred * torch.log(pred)
        entropy_loss = self.beta * torch.sum(lik2, dim= -1).mean()

        return (Advantage_loss + entropy_loss)

    def get_action(self, obs):
        with torch.no_grad():
            probs, _ = self.localnet(obs)
            probs = torch.clip(probs, 1e-8, 1-1e-8)
        probs = probs.numpy()[0]

        action = np.random.choice(self.action_space, p = probs)
        return action

    def run(self):
        total_step = 0
        while self.G_episode.value < self.rule.max_episode:
            obs_frag = self.env.reset()
            obs_frag = self.rule.preprocessing(obs_frag)
            for _ in range(4): self.obs_tmp.append(obs_frag)
            score = 0
            while True:
                obs = self.get_obs()
                total_step += 1
                action = self.get_action(obs)
                reward = 0
                for _ in range(self.rule.frame_skips):
                    if self.rule.RENDER and self.id == 0:
                        self.env.render()
                    obs_frag_, reward_, done, _ = self.env.step(action+2)
                    reward += reward_
                    if done:
                        break
                reward /= self.rule.frame_skips
                obs_frag_ = self.rule.preprocessing(obs_frag_)
                self.obs_tmp.append(obs_frag_)
                score += reward
                self.buffer_s.append(obs)
                self.buffer_a.append(action)
                self.buffer_r.append(reward)
    
                if total_step % self.rule.update_cycle == 0 or done:
                    obs_ = self.get_obs()
                    self.update_global(obs_, done)
                    self.sync()
                    if done:
                        self.record(score)
                        break
                

        self.score_queue.put(None)
    
    def update_global(self, s_, done):
        self.optimizer.zero_grad()
        S = torch.from_numpy(np.vstack(self.buffer_s)).float()
        onehot_actions = [np.eye(self.rule.action_dim)[i] for i in self.buffer_a]
        A = torch.from_numpy(np.array(onehot_actions))
        R = torch.from_numpy(np.array(self.buffer_r)).float().view(-1,1)
        D = torch.Tensor([done]).bool()
        s_ = torch.Tensor(s_).float()

        probs, values = self.localnet(S)

        if done:
            values_ = 0
        else:
            _, values_ = self.localnet(s_)

        critic_target = torch.zeros_like(R)
        running_add = values_
        for i in reversed(range(len(critic_target))):
            critic_target[i] = R[i] + self.gamma * running_add
            running_add = critic_target[i]
    
        critic_loss = self.critic_loss_fn(values, critic_target)
        
        Advantage = critic_target - values
        
        actor_loss = self.actor_loss_fn(probs, A, Advantage)
        total_loss = (actor_loss + critic_loss) / 2
        total_loss.backward()

        ''' How to calculate Actor Loss
        l1 = torch.sum(probs[0] * A[0]) * ciritc_target[0:]/self.gamma**0 - values[0]
        l2 = torch.sum(probs[1] * A[1]) * ciritc_target[1:]/self.gamma**1 - values[1]
        l3 = torch.sum(probs[2] * A[2]) * ciritc_target[2:]/self.gamma**2 - values[2]
        ...
        entropy_loss1 = torch.sum(probs[0] * log(probs[0]))
        entropy_loss2 = torch.sum(probs[1] * log(probs[1]))
        entropy_loss3 = torch.sum(probs[2] * log(probs[2]))
        ...
        action loss = SUM all above. '''   

        for l_params, g_params in zip(self.localnet.parameters(),
                                         self.globalnet.parameters()):
            g_params._grad = l_params.grad

        self.optimizer.step()

        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    def sync(self):
        self.localnet.load_state_dict(self.globalnet.state_dict())

    def record(self, score):
        with self.G_episode.get_lock():
            self.G_episode.value += 1
        with self.G_episode_score.get_lock():
            if self.G_episode_score.value == 0.:
                self.G_episode_score.value = score
            else:
                self.G_episode_score.value = self.G_episode_score.value * 0.99 + score * 0.01
        self.score_queue.put(self.G_episode_score.value)
        print(
            self.name,
            "Episode:", self.G_episode.value,
            "|MA Score: %.0f" % self.G_episode_score.value,
        )

    def get_obs(self):
        obs = torch.cat((self.obs_tmp[0],
                            self.obs_tmp[1],
                            self.obs_tmp[2],
                            self.obs_tmp[3]),
                            dim = 1)
        return obs
                            
                            