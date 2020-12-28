import gym
import torch
from torch.optim import Adam
import torch.multiprocessing as mp
from ACNetwork import ActorCriticNetwork
from Agent import ACAgent
from Rule import Rule
from SharedAdam import SharedAdam
import matplotlib.pyplot as plt

class GlobalAgent:
    def __init__(self):
        self.env_num = mp.cpu_count()
        self.Rule = Rule()
        self.globalnet = ActorCriticNetwork(self.Rule.obs_dim, self.Rule.action_dim, self.Rule.fc1_dim, self.Rule.fc2_dim)
        if self.Rule.load:
            self.globalnet.load_state_dict(torch.load(self.Rule.path))
        self.globalnet.share_memory()
        self.optimizer = SharedAdam(self.globalnet.parameters(), lr=self.Rule.lr, betas=(0.92, 0.999))
        self.hire()

    def hire(self):
        self.Agents = [ACAgent(self.globalnet, self.optimizer, i, self.Rule) for i in range(self.env_num)]

    def start(self):
        [agent.start() for agent in self.Agents]
        score_list = []
        while True:
            score = self.Rule.score_queue.get()
            if score is not None:
                score_list.append(score)
            else:
                torch.save(G_agent.globalnet.state_dict(), self.Rule.path)
                break
        [agent.join() for agent in self.Agents]

        plt.plot(score_list)
        plt.ylabel('Moveing Average Score')
        plt.xlabel('Step')
        plt.show()


if __name__ == '__main__':
    G_agent = GlobalAgent()
    G_agent.start()
    
