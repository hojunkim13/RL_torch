import sys
sys.path.append('c:\\Users\\KHJ\\Desktop\\deeplearn\\Reinforcement\\torch\\2048-Project')
from PolicyIteration.Network import Network
from torch.optim import Adam
from Environment.Utils import *
from MCTS_UCT_Valuenet import MCTS
import torch
import numpy as np
from Logger import logger
import time



class Agent:
    def __init__(self, state_dim, action_dim, lr, batch_size, n_sim):
        self.net = Network(state_dim, action_dim)
        self.optimizer = Adam(self.net.parameters(), lr = lr, weight_decay= 1e-4, betas=(0.8, 0.999))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
                
        self.n_sim = n_sim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.memory = []
        self.mcts = MCTS(self.net)

    def getAction(self, log):
        start_time = time.time()
        
        action = self.mcts.search(self.n_sim)
                
        if log:
            for line in self.mcts.root_grid:
                logger.info(f"{line}")        
            time_spend = time.time() - start_time
            act_dir = {0:"LEFT", 1:"UP",2:"RIGHT",3:"DOWN"}[int(action)]
            logger.info(f"# Step {self.step_count}, : {act_dir}, Thinking Time : {time_spend:.1f}sec")
            logger.info(f"# Action : {act_dir}, N : {self.mcts.root_node.N.values}\n\n")            

        self.step_count += 1
        return action

    def storeTransition(self, *transition):
        state = preprocessing(transition[0])
        self.memory.append(state)

    def learn(self, outcome):
        memory = np.array(self.memory, dtype = np.float32)
        S = torch.tensor(memory, dtype = torch.float).cuda().reshape(-1, *self.state_dim)
                
        _, value = self.net(S)
        outcome = torch.tensor(outcome, dtype = torch.float).cuda()
        value_loss = torch.square(value - outcome).mean()
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        self.memory = []
        return value_loss.item()

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./data/model/{env_name}_2048zero.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./data/model/{env_name}_2048zero.pt")
        self.net.load_state_dict(state_dict)