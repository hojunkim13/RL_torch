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
    def __init__(self, parent, move, grid = None):
        self.parent = parent
        self.W = 0
        self.N = 0
        self.child = {}
        self.move = move
        if grid is None:
            self.legal_moves = [1,1,1,1]
        else:
            self.legal_moves = get_legal_moves(grid)
        self.untried_moves = self.legal_moves.copy()

    def calcUCT(self, c_uct = 0.3):
        Q = self.W / self.N
        exp_component = c_uct * np.sqrt(np.log(self.parent.N) / self.N)
        return Q + exp_component        
        
    def isLeaf(self):
        return self.untried_moves != []

    def isRoot(self):
        return self.parent is None
        
class MCTS:
    def selection(self):
        node = self.root_node
        while not node.isLeaf():            
            node = max(node.child.values(), key = Node.calcUCT)
        return node

    def expansion(self, node):        
        move = np.random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        child_node = Node(node, move)
        node.child[move] = child_node
        return child_node

    def evaluation(self, target_node):
        '''
        1. Move to leaf state fow action history
        2. Start simulation from leaf state
        3. Calc average score from terminal gridsoll
        '''                    
        grid = self.root_grid
        node = target_node

        move_history = []
        while not node.isRoot():
            move_history.append(node.move)
            node = node.parent
            
        for move in move_history:
            grid = move_grid(grid, move)

        #rollout        
        while not isEnd(grid):                    
            legal_moves = get_legal_moves(grid)
            move = np.random.choice(legal_moves)
            grid = move_grid(grid, move)
        return calc_value(grid)

    def backpropagation(self, node, value):        
        node.W += value
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)
                    
    def simulation(self):        
        leaf_node = self.selection()        
        child_node = self.expansion(leaf_node)
        value = self.evaluation(child_node)
        self.backpropagation(child_node, value)

    def getAction(self, root_grid, n_sim):                        
        self.root_grid = root_grid
        self.root_node = Node(None, None, root_grid)        
        for _ in range(n_sim):
            self.simulation()
        move = max(self.root_node.child.values(), key = lambda x : x.N).move
        return move
       

n_episode = 10
n_sim = 100
env.goal = 999999

def main():
    mcts = MCTS()
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        score = 0
        grid = env.reset()    
        while not done:        
            #env.render()
            action = mcts.getAction(grid, n_sim)
            if action not in get_legal_moves(grid):
                print("warning")
            grid, reward, done, info = env.step(action)
            score += reward        
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        spending_time = time.time() - start_time
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}")
        print(f"SPENDING TIME : {spending_time:.1f} Sec")
    env.close()

if __name__ == "__main__":
    main()
