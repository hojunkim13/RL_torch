import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from Environment.Utils import *
import time
from PolicyIteration.Network import Network
import torch

n_sim = 100
n_episode = 1000
env_name = "2048"

from Environment.DosEnv import _2048
env = _2048()

# import os
# from _2048 import Game2048
# from Environment.PrettyEnv import Game2048_wrapper
# import pygame
# p1 = os.path.join("data/game", '2048_.score')
# p2 = os.path.join("data/game", '2048_.%d.state')        
# screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
# pygame.init()
# pygame.display.set_caption("2048!")
# pygame.display.set_icon(Game2048.icon(32))
# env = Game2048_wrapper(screen, p1, p2)
# env.draw()


class MCTS:
    def __init__(self):
        self.values = np.zeros(4)        
        self.net = Network((16,4,4), 4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3)
        self.memory = []

    def setRootGrid(self, grid):
        self.root_grid = grid
        self.values = np.zeros(4)

    def simulation(self):
        legal_moves = get_legal_moves(self.root_grid)        
        for first_action in legal_moves:            
            mean_value = 0
            for _ in range(10):                
                grid = move_grid(self.root_grid, first_action)
                state = preprocessing(grid)
                state = torch.tensor(state, dtype = torch.float).cuda().unsqueeze(0)
                with torch.no_grad():
                    _, value = self.net(state)
                mean_value += value.squeeze().cpu().item()
            self.values[first_action] += mean_value
        action = np.argmax(self.values)
        self.values = np.zeros(4)
        return action        
    
    def learn(self, outcome):
        state = torch.tensor(self.memory, dtype = torch.float).view(-1, 16, 4, 4).cuda()
        _, values = self.net(state)
        outcome = torch.ones_like(values, dtype = torch.float).cuda() * outcome

        loss = torch.nn.functional.mse_loss(values, outcome)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
        


def main():
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):        
        done = False
        grid = env.reset()
        score = 0
        while not done:
            mcts.setRootGrid(grid)
            action = mcts.simulation()            
            grid, reward, done, info = env.step(action)
            mcts.memory.append(preprocessing(grid))
            score += reward
        outcome = np.log2(np.sum(grid))
        mcts.learn(outcome)   

        score_list.append(score)
        average_score = np.mean(score_list[-100:])                
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}, Max Tile : {info}")
            

if __name__ == "__main__":
    main()

