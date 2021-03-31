import gym
from torchvision import transforms


class Environment:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Grayscale(),
                                            transforms.Normalize([0.5],[0.5])
                                            ])                                        

    def preprocessing(self,state):
        state = self.transforms(state.copy())
        return state.view(-1,1,96,96)

    def step(self, action, render = False):
        reward = 0
        for _ in range(8):
            if render:
                self.env.render()
            state, tmp_reward, done, info = self.env.step(action)
            reward += tmp_reward
            if done:
                break
        reward /= 8
        new_state = self.preprocessing(state)
        state_difference = new_state - self.old_state
        self.old_state = new_state
        return state_difference, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.preprocessing(state)
        self.old_state = state
        return state

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    



