import gym
from torchvision import transforms

class Rule:
    def __init__(self):
        self.n_episode = 1000
        self.save_cycle = 10
        self.load = False
        self.render = True

        self.env_name = 'CarRacing-v0'
        self.frame_stack = 4
        self.frame_skip = 8
        #Hyperparameters for Training
        self.alpha = 1e-4
        self.beta = 1e-3
        self.gamma = 0.99
        self.tau = 5e-3

        #Hyperparameters for Environment
        self.env = gym.make(self.env_name)
        #state = (3, 96, 96) --> (4, 96, 96) (stacking)
        
        self.state_dim = (4, 32, 32)
        self.action_dim = self.env.action_space.shape[0]

        #Hyperparameters for ReplayBuffer
        self.maxlen = 500000
        self.batch_size = 256