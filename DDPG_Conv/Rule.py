import gym
from torchvision import transforms

class Rule:
    def __init__(self):
        self.n_episode = 1000
        self.save_cycle = 1
        self.load = True
        self.render = False

        self.env_name = 'CarRacing-v0'
        self.frame_stack = 3
        self.frame_skip = 4
        #Hyperparameters for Training
        self.alpha = 1e-4 #2.5e-5
        self.beta = 1e-3 #2.5e-4
        self.gamma = 0.99
        self.tau = 1e-3

        #Hyperparameters for Environment
        self.env = gym.make(self.env_name)
        #state = (3, 96, 96) --> (4, 96, 96) (stacking)
        
        self.state_dim = (3, 48, 48)
        self.action_dim = self.env.action_space.shape[0]

        #Hyperparameters for ReplayBuffer
        self.maxlen = 80000
        self.batch_size = 64