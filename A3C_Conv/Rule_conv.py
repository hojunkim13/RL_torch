import gym
import torch.multiprocessing as mp
from torchvision.transforms import transforms
from PIL import Image

class Rule:
    def __init__(self):
        self.ENV_NAME = 'Pong-v0'
        self.RENDER = True
        self.load = False
        self.max_episode = 3000
        self.update_cycle = 5
        self.gamma = 0.99
        self.lr = 1e-4
        self.fc1_dim = 128
        self.fc2_dim = 128
        self.beta = 1e-2
        self.action_dim = 3
        self.frame_skips = 4
        self.action_space = [[0.,0.,0.], #Do nothing
                             [0.,.3,0.], #Accel
                             [0.,0.,.15], #break
                             [-0.3,0.1,0.], #left
                             [+0.3,0.1,0.]] #right
        self.action_space = [i for i in range(self.action_dim)]
        self.env = gym.make(self.ENV_NAME)
        self.path = f'./model/{self.ENV_NAME}.pt'
        self.obs_dim = self.env.observation_space.shape[0]
        self.G_episode, self.G_episode_score, self.score_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        
        self.transform = transforms.Compose([transforms.Resize((80,80)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5])],
                                        )
    def preprocessing(self, obs):
        obs = Image.fromarray(obs)
        obs = self.transform(obs)
        return obs.unsqueeze(0)
