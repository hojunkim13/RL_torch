import gym
import torch
from torchvision import transforms
from collections import deque

class Environment:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Grayscale(),                                            
                                            ])
        self.state_layer = deque(maxlen = 4)

    def preprocessing(self,state):
        state = self.transforms(state.copy())
        state = state[:, 4:84, 8:88]
        return state.unsqueeze(0)

    def step(self, action, render = False):
        # """
        # input action space : (-1, -1, -1) ~ (+1, +1, +1)
        # env action space : (-1, 0, 0 ~ (+1, +1, +1)
        # So, We have to adjust input action
        # """
        #action = (action + [0, +1, +1]) * [1, 0.5, 0.5]

        reward = 0
        for _ in range(8):
            if render:
                self.env.render()
            state, tmp_reward, done, info = self.env.step(action)
            reward += tmp_reward
            if done:
                break
        state = self.preprocessing(state)
        #state_difference = new_state - self.old_state
        #self.old_state = new_state
        self.state_layer.append(state)
        new_state = torch.cat((self.state_layer[0], self.state_layer[1], self.state_layer[2], self.state_layer[3]), dim = 1)
        self.reward_history.append(reward)
        if len(self.reward_history) == 100 and max(self.reward_history) <= -0.1:
            done = True
            reward = -50
        return new_state, reward, done, info

    def reset(self):
        self.reward_history = deque(maxlen = 100)
        state = self.env.reset()
        state = self.preprocessing(state)
        for _ in range(4):
            self.state_layer.append(state)
        #self.old_state = state
        new_state = torch.cat((self.state_layer[0], self.state_layer[1], self.state_layer[2], self.state_layer[3]), dim = 1)
        return new_state

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    



