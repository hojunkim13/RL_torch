from PolicyIteration.Network import Network
from torch.optim import Adam
from Environment.Utils import *
from MyMCTS import MCTS
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from Logger import logger


class Agent:
    def __init__(self, state_dim, action_dim, lr, batch_size, n_sim):
        self.net = Network(state_dim, action_dim)
        self.optimizer = Adam(self.net.parameters(), lr = lr, weight_decay= 1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 400*1000, 0.1)
        
        self.n_sim = n_sim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.trajectory = []

    def getAction(self, grid):
        start_time = time.time()
        mcts = MCTS(grid, self.net)
        
        while mcts.search_count != self.n_sim:
            mcts.tree_search()
        tau = 1 if self.step_count < 100 else 0
        probs = mcts.get_probs(tau)
        if tau == 1:
            dist = torch.distributions.Categorical(torch.tensor(probs, dtype=torch.float).cuda())
            action = dist.sample().detach().cpu().numpy()
        else:
            action = probs.index(1)

        for line in grid:
            logger.info(line)

        act_dir = {0:"LEFT", 1:"UP",2:"RIGHT",3:"DOWN"}[int(action)]
        thinking_time = time.time() - start_time
        logger.info(f"# Action : {act_dir}, Thinking Time : {thinking_time:.1f}sec\n\n")        
        return action

    def storeTransition(self, *transition):
        state = preprocessing(transition[0])
        action = int(transition[1])
        transition = np.array((state, action), dtype = object)
        self.trajectory.append((state, action))

    def learn(self, outcome):
        try:
            trajectory = random.sample(self.trajectory, self.batch_size)        
        except ValueError:
            trajectory = self.trajectory
        trajectory = np.array(trajectory, dtype = object)
        S = np.vstack(trajectory[:,0]).reshape(-1, *self.state_dim)
        A = np.vstack(trajectory[:,1])
        Z = np.ones(A.shape) * outcome

        S = torch.tensor(S, dtype = torch.float).cuda()
        A = torch.tensor(A, dtype = torch.float).cuda()
        Z = torch.tensor(Z, dtype = torch.float).cuda()

        policy, value = self.net(S)
        value_loss = F.mse_loss(value, Z)
        policy_loss = -(A * policy.log_prob(A + 1e-8)).mean()
        total_loss = 0.6 * value_loss + 0.4 * policy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.trajectory = []

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./data/model/{env_name}_2048zero.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./data/model/{env_name}_2048zero.pt")
        self.net.load_state_dict(state_dict)