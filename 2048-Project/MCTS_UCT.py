from Environment.Utils import *
import numpy as np
import os
import time
import pickle

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

class Node:
    def __init__(self, parent):
        self.parent = parent
        self.W = [0, 0, 0, 0]
        self.N = [0, 0, 0, 0]
        self.child = [0, 0, 0, 0]

    def calcUCT(self, c_uct = 0.1):        
        Qs = [w/(n+1e-8) for w, n in zip(self.W, self.N)]
        exp_components = [c_uct * np.sqrt(np.log(sum(self.N)) / (n)) for n in self.N]
        UCT_values = [q + exp_component for q, exp_component in zip(Qs, exp_components)]
        return np.argmax(UCT_values)

    def isLeaf(self):
        return any([n == 0 for n in self.N])

    def isRoot(self):
        return self.parent is None

    def asRoot(self):
        self.parent = None

class MCTS:
    def __init__(self):
        self.Q_data = []
        self.grid_data = []

    def setRoot(self, grid, act = None):
        if act is not None:
            new_root_node = self.root_node.child[act]
            new_root_node.asRoot()
            self.root_node = new_root_node            
        else:
            self.root_node = Node(None)
        self.root_grid = grid
        

    def selection(self):
        node = self.root_node
        self.act_history = []
        while not node.isLeaf():
            child_idx = node.calcUCT()
            node = node.child[child_idx]
            self.act_history.append(child_idx)
        return node

    def expansion(self, node):
        zero_indices = []
        for idx in range(4):
            if node.child[idx] == 0:
                zero_indices.append(idx)
        expand_act = np.random.choice(zero_indices)
        child_node = Node(node)
        node.child[expand_act] = child_node
        return expand_act

    def simulation(self, expand_act, k = 10):
        '''
        1. Move to leaf state fow action history
        2. Start simulation from leaf state
        3. Calc average score from terminal gridsoll
        '''
        values = []
        for _ in range(k):
            #sim to leaf grid
            grid = self.root_grid
            for act in self.act_history:
                grid = move_grid(grid, act)

            #expanded grid
            grid = move_grid(grid, expand_act)

            #sim to terminal grid
            while not isEnd(grid):
                act = np.random.randint(0, 4)
                grid = move_grid(grid, act)
            value = calc_value(grid)
            values.append(value)

        return np.mean(values)

    def backpropagation(self, leaf_node, value, expand_act):        
        node = leaf_node
        act_history = self.act_history.copy()
        act_history.append(expand_act)
        for act in reversed(act_history):
            node.W[act] += value
            node.N[act] += 1
            node = node.parent
            
        
    def simCycle(self):        
        leaf_node = self.selection()
        expand_act = self.expansion(leaf_node)
        grid_value = self.simulation(expand_act)
        self.backpropagation(leaf_node, grid_value, expand_act)

    def getAction(self, n_sim):                
        for _ in range(n_sim):
            self.simCycle()
        act = np.argmax(self.root_node.N)
        self.storeTransition()
        return act
       
    def storeTransition(self):
        Qs = [w/(n+1e-8) for w, n in zip(self.root_node.W, self.root_node.N)]
        grid = self.root_grid
        self.Q_data.append(Qs)
        self.grid_data.append(grid)
        
    def saveMemory(self, e):
        datas = [self.grid_data, self.Q_data]
        with open("./data/transition/{e}.pkl", mode = "wb") as fo:
            pickle.dump(datas, fo)
        self.grid_data = []
        self.Q_data = []
    
n_episode = 100
n_sim = 400
env.goal = 999999

def main():
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        score = 0
        grid = env.reset()
        mcts.setRoot(grid)        
        while not done:        
            action = mcts.getAction(n_sim)
            grid, reward, done, info = env.step(action)
            score += reward
            mcts.setRoot(grid, action)            
        mcts.saveMemory(e)
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        spending_time = time.time() - start_time
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}")
        print(f"SPENDING TIME : {spending_time:.1f} Sec")
    env.close()

if __name__ == "__main__":
    main()
