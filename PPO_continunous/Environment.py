import gym
import torch
from torchvision import transforms
from collections import deque
import numpy as np


class Environment:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.state_dim = (3,96,96)
        self.action_dim = 3
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Grayscale(),
                                            transforms.Normalize([0.5],[0.5])
                                            ])
        self.tmp = deque(maxlen = 4)

    def preprocessing(self,state):
        state = torch.Tensor(state).cuda()
        state = self.transforms(state)
        return state.view(-1,1,96,96)

    def step(self, action):
        reward = 0
        for _ in range(8):
            state_, tmp_reward, done, info = self.env.step(action)
            reward += tmp_reward
        reward = np.clip(reward, -5, 5)
        state_ = self.preprocessing(state_)
        self.tmp.append(state_)
        state_ = self.get_state()
        return state_, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.preprocessing(state)
        for _ in range(4):
            self.tmp.append(state)
        return self.get_state()

    def get_state(self):
        state = np.concatenate((self.tmp[0], self.tmp[1],
                                self.tmp[2], self.tmp[3]), axis = 0)
        return state

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    



