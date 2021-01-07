import gym
from PIL import Image
from collections import deque
import numpy as np

class Environment:
    def __init__(self, env_name, frame_skip, frame_stack):
        self.env = gym.make(env_name)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.buffer = deque(maxlen = frame_stack)
        self.reward_counter = 0

    def step(self, action, render = False):
        total_reward = 0
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        for _ in range(self.frame_skip):
            if render:
                self.env.render()
            state_, reward, done, info = self.env.step(action)
            
            if np.mean(state_[:, :, 1]) > 185.0:
                reward -= 0.05
            if done:
                break

            total_reward += reward
        state_ = self.preprocessing(state_)
        self.buffer.append(state_)
        state_ = self.get_state()
        return state_, total_reward, done, info

    def preprocessing(self, state):
        # 3 * 96 * 96   >>1 * 1 * 96 * 96
        state = Image.fromarray(state, 'RGB').convert('L')
        state = np.asarray(state)
        state = state / 255.0
        state = np.reshape(state,(1,1,96,96))
        return state

    def reset(self):
        self.reward_counter = 0
        state = self.env.reset()
        state = self.preprocessing(state)
        for _ in range(self.frame_stack):
            self.buffer.append(state)
        state = self.get_state()
        return state

    def get_state(self):
        state = np.concatenate((self.buffer[0], self.buffer[1],
                                self.buffer[2], self.buffer[3]), axis = 1)
        return state


