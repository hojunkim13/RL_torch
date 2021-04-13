#Monte Carlo Tree search for 2048 games
from Environment.Utils import *
from Environment import logic
import torch


action_space = [logic.move_left,
                logic.move_up,
                logic.move_right,
                logic.move_down]

def get_legal_moves(grid):
    legal_moves = []
    for act_func in action_space:
        if act_func(grid)[1]:
            legal_moves.append(1)
        else:
            legal_moves.append(0)                        
    return legal_moves

class Node:
    def __init__(self, grid, parent):        
        self.grid = grid
        self.parent = parent
        self.legal_moves = get_legal_moves(grid)
        #self.edges = self.get_edge(actor, critic)
        self.children = []
        
    def get_edge(self, actor):
        edges = []
        for move_idx in range(len(self.legal_moves)):
            if self.legal_moves[move_idx] == 0:
                continue
            else:
                new_grid = action_space[move_idx](self.grid)
                self.children.append(Node(new_grid, parent = self))
            with torch.no_grad():
                new_state = preprocessing(new_grid)
                probability = actor(new_state)[0].detach().cpu().numpy()
            edge = Edge(parent_node = self, probability = probability)
            edges.append(edge)


class Edge:
    def __init__(self, parent_node, probability):
        self.parent_node = parent_node
        self.status = {
                    "Q" : 0,
                    "N" : 0,
                    "P" : probability
        }
        pass


class MCTS:
    def __init__(self, root_grid):
        self.root_node = Node(root_grid, None)

    def select(self):
        pass

    def expand(self):
        pass

    def evaluate(self):
        pass

    def backup(self):
        pass

    def succeed(self):
        pass
