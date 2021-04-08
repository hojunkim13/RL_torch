import gym
from torchvision import transforms


class Environment:
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Grayscale(),                                            
                                            ])                                        

    def preprocessing(self,state):
        state = self.transforms(state.copy())
        state = state[:, :84, 6:90]
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
    



