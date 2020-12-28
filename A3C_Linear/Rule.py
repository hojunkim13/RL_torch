import gym
import torch.multiprocessing as mp

class Rule:
    def __init__(self):
        self.ENV_NAME = 'LunarLander-v2'
        self.RENDER = True
        self.load = False
        self.max_episode = 3000
        self.update_cycle = 5
        self.gamma = 0.99
        self.lr = 1e-4
        self.fc1_dim = 128
        self.fc2_dim = 128
        self.beta = 1e-2
        self.env = gym.make(self.ENV_NAME)
        self.path = f'./model/{self.ENV_NAME}.pt'
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.G_episode, self.G_episode_score, self.score_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()