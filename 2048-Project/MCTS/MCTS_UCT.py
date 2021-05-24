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


class Node:
    def __init__(self, parent, move, legal_moves=[0, 1, 2, 3]):
        self.parent = parent
        self.W = 0
        self.N = 0
        self.child = {}
        self.move = move
        self.legal_moves = legal_moves
        self.untried_moves = self.legal_moves.copy()

    def calcUCT(self, c_uct=0.8):
        Q = self.W / self.N
        exp_component = c_uct * np.sqrt(np.log(self.parent.N) / self.N)
        return Q + exp_component

    def isLeaf(self):
        return self.untried_moves != []

    def isRoot(self):
        return self.parent is None

class MCTS:
    def __init__(self):
        self.last_move = None

    def getDepth(self, node):
        depth = 0
        while not node.isRoot():
            depth += 1
            node = node.parent
        return depth

    def select(self, node, grid):
        if node.isLeaf():
            return node, grid
        else:
            node = max(node.child.values(), key = Node.calcUCT)
            grid = move_grid(grid, node.move)
            return self.select(node, grid)

    def expand(self, node):
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        child_node = Node(node, move)
        node.child[move] = child_node
        return child_node

    def evaluate(self, grid):                        
        # if len(free_cells(grid)) >= 5 and np.max(grid) <= 1024:
        #     max_step = 80
        # else:
        #     max_step = 80
        max_step = 80
        merged_sum = 0        
        step = 0
        while not isEnd(grid):
            grid, merged_val = self.SNM_policy(grid)
            merged_sum += merged_val
            step += 1
            if step >= max_step:
                break
        return merged_sum

    def SNM_policy(self, grid):
        snm_counts = []        
        for move in range(2):
            _, _, merged_sum = move_and_get_sum(grid, move)
            snm_counts.append(merged_sum)            

        if snm_counts[0] >= snm_counts[1]:
            move = random.choice([0, 2])            
        else:
            move = random.choice([1, 3])

        grid = move_grid(grid, move)
        return grid, max(snm_counts)

    def CNM_policy(self, grid):
        cnm_counts = []
        for move in range(0, 1):
            grid_ = move_grid(grid, move)
            cnm_counts.append(len(free_cells(grid_)))
        
        if cnm_counts[0] >= cnm_counts[1]:
            move = random.choice([0, 2])            
        else:
            move = random.choice([1, 3])
        grid = move_grid(grid, move)
        return grid

    def backpropagation(self, node, value):
        node.W = (node.N * node.W + value) / (node.N + 1)
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)

    def searchTree(self):
        node = self.root_node
        grid = self.root_grid
        leaf_node, grid = self.select(node, grid)   

        if not isEnd(grid):
            child_node = self.expand(leaf_node)
            value = self.evaluate(grid)            
            self.backpropagation(child_node, value)
        else:
            self.backpropagation(leaf_node, 0)

    def getAction(self, root_grid, n_sim):
        # self.root_node = Node(None, None, get_legal_moves(root_grid))
        # self.root_grid = root_grid
        
        if self.last_move is None:
            self.root_node = Node(None, None, get_legal_moves(root_grid))
            self.root_grid = root_grid
        else:
            new_root_node = self.root_node.child[self.last_move]
            new_root_node.parent = None
            new_root_node.move = None

            new_root_node.legal_moves = get_legal_moves(root_grid)            
            unlegal_moves = list(set([0, 1, 2, 3]) - set(new_root_node.legal_moves))
            for move in unlegal_moves:
                del new_root_node.child[move]
            self.root_node = new_root_node
            self.root_grid = root_grid
        
        for _ in range(n_sim):            
            self.searchTree()
        
        robust_move = max(self.root_node.child.values(), key=lambda x: x.N).move
        self.last_move = robust_move
        return robust_move
    
        


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
            if action not in get_legal_moves(grid):
                raise SystemError("Agent did a unlegal action")
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
