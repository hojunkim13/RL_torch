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

    def succeed(self, grid):
        if grid in self.states:
            self.states = [grid]
            self.parent = None
            self.move = None
            self.legal_moves = get_legal_moves(grid)
            unlegal_moves = list(set([0, 1, 2, 3]) - set(self.legal_moves))
            for move in unlegal_moves:
                del self.child[move]
            return True
        else:
            return False


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
        if len(free_cells(grid)) <= 5:
            max_step = 80
        else:
            max_step = 40

        step = 0
        merged_sum = 0

        while not isEnd(grid):
            move = self.CNM_policy(grid)
            #move = random.choice(range(4))
            grid, _, merged_val = move_and_get_sum(grid, move)
            merged_sum += merged_val
            step += 1
            if step >= max_step:
                break        
        return merged_sum

    def SNM_policy(self, grid):
        snm_counts = []
        for move in range(4):
            _, _, merged_sum = move_and_get_sum(grid, move)
            snm_counts.append(merged_sum)
        return np.argmax(snm_counts)

    def CNM_policy(self, grid):
        cnm_counts = []
        for move in range(4):
            grid_ = move_grid(grid, move)
            cnm_counts.append(len(free_cells(grid_)))
        return np.argmax(cnm_counts)

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
        self.root_node = Node(None, None, get_legal_moves(root_grid))
        self.root_grid = root_grid
        
        # if self.last_move is None:
        #     self.root_node = Node(None, None, get_legal_moves(root_grid))
        #     self.root_node.states = [root_grid]
        # else:
        #     self.root_node = self.reuseTree(root_grid)

        for _ in range(n_sim):
            self.searchTree()

        # move = max(self.root_node.child.values(), key = lambda x : x.W / x.N).move
        move = max(self.root_node.child.values(), key=lambda x: x.N).move
        self.last_move = move
        return move

    def reuseTree(self, root_grid):
        subtree = self.root_node.child[self.last_move]
        success = subtree.succeed(root_grid)
        if success:
            return subtree
        else:
            node = Node(None, None, get_legal_moves(root_grid))
            node.states = [root_grid]
            return node


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
    main(n_episode=1, n_sim=200)
