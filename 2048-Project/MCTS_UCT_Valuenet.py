from Environment.Utils import *
import numpy as np
import os
import torch
import time

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
    def __init__(self, parent, move_index):
        self.parent = parent
        self.move_index = move_index
        self.W = {0:0, 1:0, 2:0, 3:0}
        self.N = {0:0, 1:0, 2:0, 3:0}
        self.child = {}
        self.legal_moves = [1,1,1,1]

    def calcUCT(self, c_uct = 0.5):                
        UCT_values = {}
        Qs = {}
        EXPs = {}
        N_total = sum(self.N.values())
        for idx in self.child.keys():
            if not self.legal_moves[idx]:
                continue
            w = self.W[idx]
            n = self.N[idx]
            Qs[idx] = w/n            
            EXPs[idx] = c_uct * np.sqrt(np.log(N_total) / n)
        
        for (idx, Q), exp in zip(Qs.items(), EXPs.values()):
            UCT_values[idx] = Q + exp
        return UCT_values

    def isLeaf(self):        
        return sum(self.legal_moves) != len(self.child)

    def isRoot(self):
        return self.parent is None

    def asRoot(self):
        self.parent = None
        self.move_index = None
        
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_index)
            self.parent.getPath(path_list)
        

class MCTS:
    def __init__(self, net):
        self.net = net

    def reset(self, grid):
        self.root_node = Node(None, None)
        self.root_grid = grid
        self.root_node.legal_moves = get_legal_moves(grid)


    def setRoot(self, grid, act): 
        self.root_node = Node(None, None)
        self.root_grid = grid
        self.root_grid.legal_moves = get_legal_moves(grid)
        # new_root_node = self.root_node.child[act]
        # new_root_node.asRoot()

        # #self.root_node = new_root_node            
        # self.root_grid = grid
        # self.root_node.legal_moves = get_legal_moves(grid)
        
        # for idx in range(4):
        #     if not self.root_node.legal_moves[idx]:
        #         self.root_node.N[idx] = 0
        #         self.root_node.W[idx] = 0
        #         try:
        #             del self.root_node.child[idx]
        #         except KeyError:
        #             pass
        
            

    def selection(self, node):
        if node.isLeaf():
            return node
        else:
            UCT_values = node.calcUCT()            
            max_value_idx = max(UCT_values, key=UCT_values.get)
            
            node = node.child[max_value_idx]
            return self.selection(node)
        

    def expansion(self, node):                
        for idx, n in node.N.items():
            if n == 0:
                break
        child_node = Node(node, idx)
        node.child[idx] = child_node
        return child_node

    def simulation(self, child_node, k = 10):
        '''
        1. Move to leaf state follow action history
        2. Start simulation from leaf state
        3. Calc average score via value network
        '''
        states = []
        act_history = []
        #vs = []
        child_node.getPath(act_history)
        
        for _ in range(k):
            #sim to child grid
            grid = self.root_grid            
            for act in act_history:
                grid = move_grid(grid, act)                        
            state = preprocessing(grid)            
            states.append(state)
            # v = self.rollout(grid)
            # vs.append(v)
            
        state_batch = torch.tensor(states, dtype = torch.float).cuda().view(-1,16,4,4)
        with torch.no_grad():
            _, values = self.net(state_batch)            
            values = torch.clip(values, 0, None)
        mean_value = values.mean().cpu().item()        
        #mean_value = np.mean(vs)
        return mean_value

    def rollout(self, grid):
        while not isEnd(grid):
            action = np.random.randint(0, 4)
            grid = move_grid(grid, action)
        value = calc_value(grid)
        return value
                

    def backpropagation(self, node, value):                
        if node.parent is not None:
            node.parent.W[node.move_index] += value
            node.parent.N[node.move_index] += 1
            self.backpropagation(node.parent, value)
                
    def search_cycle(self):
        leaf_node = self.selection(self.root_node)
        expanded_node = self.expansion(leaf_node)
        expanded_value = self.simulation(expanded_node)
        self.backpropagation(expanded_node, expanded_value)

    def search(self, n_sim):
        for _ in range(n_sim):
            self.search_cycle()
        #Max-Robust child
        while True:
            UCT_values = self.root_node.calcUCT()
            max_value_idx = max(UCT_values, key=UCT_values.get)
            max_visit_idx = max(self.root_node.N, key = self.root_node.N.get)
            if max_value_idx == max_visit_idx:
                break
            else:
                self.search_cycle()
        return max_visit_idx
       
    


def main():
    n_episode = 100
    n_sim = 400
    env.goal = 999999
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
