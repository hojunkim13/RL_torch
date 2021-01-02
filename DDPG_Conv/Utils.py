import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from collections import deque


class ReplayBuffer:
    def __init__(self, rule):
        self.batch_size = rule.batch_size
        self.maxlen = rule.maxlen
        self.S = torch.zeros((rule.maxlen, rule.frame_stack, rule.state_dim[1], rule.state_dim[2])).float()
        self.A = np.zeros((rule.maxlen, rule.action_dim), dtype='float')
        self.R = np.zeros((rule.maxlen, 1), dtype = 'float')
        self.S_ = torch.zeros((rule.maxlen, rule.frame_stack, rule.state_dim[1], rule.state_dim[2])).float()
        self.D = np.zeros((rule.maxlen, 1), dtype='bool')
        self.mem_counter = 0

    def store(self, s, a, r, s_, d):
        idx = self.mem_counter % self.maxlen
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = [r]
        self.S_[idx] = s_
        self.D[idx] = [d]
        self.mem_counter += 1

    def get_samples(self):
        length = min(self.mem_counter, self.maxlen)
        indices = np.random.choice(np.arange(length), size=self.batch_size)
        S = self.S[indices]
        A = self.A[indices]
        R = self.R[indices]
        S_ = self.S_[indices]
        D = self.D[indices]
        return S, A, R, S_, D


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)


class Tools:
    def __init__(self, rule):
        self.tmp_states = deque(maxlen=rule.frame_stack)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Grayscale(),
                                             #transforms.Resize((32, 32)), 
                                             transforms.Normalize([0.5], [0.5]),
                                             ])

    def preprocessing_image(self, state):
        state = state[:84,6:-6,:].copy()
        state = self.transform(state.copy())
        return state

    def init_weights(self, params):
        if type(params) == nn.Conv2d or type(params) == nn.Linear:
            params.weight.data.uniform_(-3e-3, 3e-3)
            params.bias.data.uniform_(-3e-4, 3e-4)

    def add_to_tmp(self, state):
        self.tmp_states.append(state)

    def get_state(self):
        state = torch.cat(
            (self.tmp_states[0], self.tmp_states[1], self.tmp_states[2], self.tmp_states[3]), dim=0).cuda()
        return state.unsqueeze(0)
