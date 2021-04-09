import gym
from gym.wrappers import Monitor
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, env_name, record = False):
        self.env = gym.make(env_name)
        if record:
            self.env = Monitor(self.env, './video', force=True)
        self.transforms = transforms.Compose([transforms.ToTensor(),                                            
                                            transforms.Grayscale(),
                                            transforms.Resize(80),
                                            ])
        

    def imgshow(self, img):
        if type(img) is torch.Tensor:
            img = img.detach().cpu().numpy()[0]
            img = np.transpose(img, (1,2,0))
            plt.imshow(img, cmap = "gray")
        else:
            img = np.array(img)
        plt.show()


    def preprocessing(self,state):
        state = state[35:195]
        state = self.transforms(state.copy())
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
    



