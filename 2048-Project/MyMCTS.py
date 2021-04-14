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
    def __init__(self, grid):
        self.grid = grid
        self.legal_moves = get_legal_moves(grid)        
        self.edges = [0, 0, 0, 0]
        self.children = [0, 0, 0, 0]
        
    def isLeaf(self, move_idx):
        if self.children[move_idx] == 0:
            return True
        else:
            return False


class Edge:
    def __init__(self, probability):        
        self.status = {
                    "N" : 0,
                    "W" : 0,
                    "Q" : 0,
                    "P" : probability
        }
        


class MCTS:
    def __init__(self, root_grid, actor, critic):
        self.root_node = Node(root_grid)
        self.tau = .1
        self.cpuct = 1
        self.search_num = 100

        self.search_count = 0
        self.actor = actor
        self.critic = critic
        state = preprocessing(root_grid)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
        probs = self.actor(state)[0].detach().cpu().numpy()
        for move_idx in range(4):
            self.root_node.edges[move_idx] = Edge(probs[move_idx])

    def select(self, node):
        Q_values = []
        P_values = []
        N_values = []
        for edge in node.edges:
            Q_values.append(edge.status["Q"])
            P_values.append(edge.status["P"])
            N_values.append(edge.status["N"])
        U_values = self.cpuct * np.array(P_values) * np.square(sum(N_values)) / (1 + np.array(N_values))

        values = [q+u for q,u in zip(Q_values, U_values)]
        if sum(values) == 0:
            move_idx = np.random.randint(0,4)
        else:
            move_idx = np.argmax(values)
        return move_idx


    def expand(self, node, move_idx):
        new_grid, _ = action_space[move_idx](node.grid)
        new_node = Node(new_grid)
        node.children[move_idx] = new_node

        state = preprocessing(new_grid)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
        probs = self.actor(state)[0].detach().cpu().numpy()
        for move_idx in range(4):
            new_node.edges[move_idx] = Edge(probs[move_idx])
        return new_node

    def evaluate(self, node):
        state = preprocessing(node.grid)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
        with torch.no_grad():
            value = self.critic(state)[0].cpu().numpy()
        return value

    def backup(self, history, value):
        for edge in reversed(history):
            edge.status["N"] += 1
            edge.status["W"] += value
            edge.status["Q"] = edge.status["W"] / edge.status["N"]

    def succeed(self):
        pass

    def tree_search(self):
        node = self.root_node
        edge_history = []
        #go to leaf
        while True:
            move_idx = self.select(node)
            if node.isLeaf(move_idx):            
                break
            else:
                edge_history.append(node.edges[move_idx])
                node = node.children[move_idx]
            
        #expand
        new_node = self.expand(node, move_idx)
        
        #evaluate
        value = self.evaluate(new_node)

        #backup
        self.backup(edge_history, value)

        self.search_count += 1

    def get_probs(self):
        N_values = []
        N_total = 0
        for edge in self.root_node.edges:
            N_values.append(edge.status["N"])
            N_total += edge.status["N"]
        probs = (np.array(N_values) / N_total) ** (1/self.tau)
        return probs
