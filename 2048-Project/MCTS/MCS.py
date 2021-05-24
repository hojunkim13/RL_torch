import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Environment.Utils import *
import numpy as np
import time
import random
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

from Environment.DosEnv import _2048

env = _2048()
env.goal = 999999


class MCTS:
    def __init__(self):        
        pass

    def SNM_policy(self, grid):
        snm_counts = []
        grids = []
        for move in range(4):
            grid_, _, merged_sum = move_and_get_sum(grid, move)
            snm_counts.append(merged_sum)
            grids.append(grid_)
        move = np.argmax(snm_counts)
        return grids[move], snm_counts[move]

    def getAction(self, root_grid, n_sim):        
        legal_moves = get_legal_moves(root_grid)
        values = {k:0 for k in legal_moves}
        visits = {k:0 for k in legal_moves}
        for _ in range(n_sim):
            action = random.choice(legal_moves)
            grid = move_grid(root_grid, action)
            step = 0
            while not isEnd(grid):
                grid, value = self.SNM_policy(grid)
                values[action] += value
                #grid = move_grid(grid, random.choice(range(4)))
                #grid = move_grid(grid, random.choice(range(4)))
                step += 1
                if step >= 80:
                    break
            # values[action] += np.sum(grid)
            visits[action] += 1

            #values[action] += np.sum(grid)
        #action = max(legal_moves, key = lambda x : values[x] / visits[x])        
        vs = [0,0,0,0]
        for idx in values.keys():
            vs[idx] = values[idx] / visits[idx]
        return np.argmax(vs)
    
        

def main(n_episode, n_sim):
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        score = 0
        grid = env.reset()
        while not done:
            env.render()
            action = mcts.getAction(grid, n_sim)
            grid, reward, done, info = env.step(action)
            score += reward
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        spending_time = time.time() - start_time
        print(
            f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}"
        )
        print(f"SPENDING TIME : {spending_time:.1f} Sec\n")
    env.close()


if __name__ == "__main__":
    main(n_episode=1, n_sim=100)
