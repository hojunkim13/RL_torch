import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Environment.Utils import *
import numpy as np
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
    def __init__(self, parent, move_index, grid):        
        self.parent = parent
        self.move_index = move_index
        self.W = 0
        self.N = 0
        self.child = {}
        self.legal_moves = get_legal_moves(grid)
        for move in self.legal_moves:
            self.child[move] = {}
        self.untried_moves = self.legal_moves.copy()
        self.grid = grid

    def calcUCT(self, c_uct = 3):
        Q = self.W / self.N
        EXP = np.sqrt(c_uct * np.log(self.parent.N) / self.N)        
        return Q + EXP

    def isLeaf(self):        
        return self.untried_moves != [] or isEnd(self.grid)
                
    def isRoot(self):
        return self.parent is None
   
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_index)
            self.parent.getPath(path_list)
        

class MCTS:
    def __init__(self, net = None):
        self.net = net
    
    def selection(self, node):
        if node.untried_moves != []:
            max_move = node.untried_moves[0]
            node.untried_moves.remove(max_move)
            return max_move
        values = {k:0 for k in node.legal_moves}
        for move in node.legal_moves:
            childs = node.child[move]            
            for child in childs.values():
                values[move] += child.calcUCT() 
            values[move] /= len(childs)
        max_move = max(values, key = values.get)
        return max_move
        
    # def expansion(self, node):
    #     if node.untried_moves != []:
    #         move = np.random.choice(node.untried_moves)
    #         node.untried_moves.remove(move)
    #     else:
    #         move = np.random.choice(node.legal_moves)                    
    #     new_grid = move_grid(node.grid, move)
    #     child_node = Node(node, move, new_grid)
    #     node.child[move].append(new_grid)
    #     return child_node

    def evaluation(self, child_node):                                
        grid = child_node.grid
        value = 0
        for _ in range(10):
            value += self.rollout(grid)
        return value / 10

    def rollout(self, grid):
        while not isEnd(grid):
            action = np.random.randint(0, 4)
            grid= move_grid(grid, action)
        value = calc_value(grid)
        return value
                
    def backpropagation(self, node, value):
        node.W += value
        node.N += 1
        if not node.isRoot():                            
            self.backpropagation(node.parent, value)
                
    def search_cycle(self):
        node = self.root_node        
        while True:
            if isEnd(node.grid):
                new_node = node
                break
            max_move = self.selection(node)
            new_grid = move_grid(node.grid, max_move)
            if str(new_grid) in node.child[max_move].keys():                
                node = node.child[max_move][str(new_grid)]                
            else:                    
                new_node = Node(node, max_move, new_grid)
                node.child[max_move][str(new_grid)] = new_node                
                break
                            
        value = self.evaluation(new_node)            
        self.backpropagation(new_node, value)

    def search(self, n_sim, grid):
        self.root_node = Node(None, None, grid)
        for _ in range(n_sim):
            self.search_cycle()
        #Max-Robust child        
        visits = {}
        for move in self.root_node.legal_moves:
            visit = 0
            childs_dict = self.root_node.child[move]
            for child in childs_dict.values():
                visit += child.N
            visits[move] = visit                
        max_move = max(visits, key = visits.get)
        return max_move
       
    
def main(n_episode, n_sim):
    env.goal = 999999
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        score = 0
        grid = env.reset()    
        while not done:
            #env.render()
            action = mcts.search(n_sim, grid)
            grid, reward, done, info = env.step(action)
            score += reward                    
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        spending_time = time.time() - start_time
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}")
        print(f"SPENDING TIME : {spending_time:.1f} Sec")
    env.close()

if __name__ == "__main__":
    n_episode = 1
    n_sim = 100
    main(n_episode, n_sim)
    
