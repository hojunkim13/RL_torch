import gym

class Rule:
    def __init__(self):
        self.epochs = 1000
        #self.env_name = 'MountainCarContinuous-v0'
        self.env_name = 'LunarLanderContinuous-v2'

        #Hyperparameters for Training
        self.alpha = 2.5e-5
        self.beta = 2.5e-4
        self.gamma = 0.99
        self.tau = 1e-3
        self.noise = 0.1

        #Hyperparameters for Environment
        
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.Load = False
        self.Save = False

        #Hyperparameters for ReplayBuffer
        self.maxlen = 1000000
        self.batch_size = 64
        
        #Nfor Network
        self.fc1_dim = 400
        self.fc2_dim = 300

        