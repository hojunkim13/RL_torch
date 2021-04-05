from collections import deque
import gym
from torchvision import transforms
import torch

class Environment:
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.transforms = transforms.Compose([transforms.ToTensor(),                                            
                                            transforms.Grayscale(),                                        
                                            ])
        


    def preprocessing(self,state):
        state = self.transforms(state.copy())
        state = state[:,35:195,:]
        state = transforms.Resize(84)(state)
        return state.unsqueeze(0)

    def step(self, action, render = False):
        if render:
            self.render()
        state, reward, done, info = self.env.step(action)
        state = self.preprocessing(state)
        state_difference = state - self.old_state
        self.old_state = state
        return state_difference, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.preprocessing(state)        
        self.old_state = state
        state_difference = state - self.old_state
        return state_difference

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    



