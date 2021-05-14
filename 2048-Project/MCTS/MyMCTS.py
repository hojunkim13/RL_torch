#Monte Carlo Tree search for 2048 games
from Environment.Utils import *
import torch

class Node:
    def __init__(self, grid, parent):
        self._parent = parent
        self.grid = grid
        self.legal_moves = get_legal_moves(grid)        
        self.children = []
        self.edges = []
        
    def isLeaf(self):
        return len(self.children) == 0

    def isRoot(self):
        return self._parent is None

class Edge:
    
    def __init__(self, probability):      
        self.status = {
                    "N" : 0,
                    "W" : 0,
                    "Q" : 0,
                    "P" : probability
        }
        

class MCTS:
    search_count = 0
    cpuct = 5
    def __init__(self, root_grid, network):
        self.root_node = Node(root_grid, None)
        self.net = network


    def _select(self, node):            
        Q_values = []
        P_values = []
        N_values = []
        for edge in node.edges:
            Q_values.append(edge.status["Q"])
            P_values.append(edge.status["P"])
            N_values.append(edge.status["N"])

        U_values = self.cpuct * np.array(P_values) * np.sqrt(sum(N_values)) / (1 + np.array(N_values))

        values = [Q+U for Q,U in zip(Q_values, U_values)]        
        idx = np.argmax(values)
        child = node.children[idx]
        return child


    def _expand_and_evaluate(self, node):
        leaf_state = preprocessing(node.grid)
        leaf_state = torch.tensor(leaf_state, dtype = torch.float).unsqueeze(0).cuda()
        with torch.no_grad():
            probs = self.net(leaf_state)
            value = calc_value(node.grid)
        probs = probs.cpu().numpy()[0]
        if node.isRoot():
            probs = 0.75 * np.array(probs) + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                
        for move_idx in range(4):
            if not node.legal_moves[move_idx]:
                continue
            new_grid, _ = action_space[move_idx](node.grid)            
            new_grid = spawn_new(new_grid)
                
            new_node = Node(new_grid, parent = node)
            new_edge = Edge(probs[move_idx])

            node.children.append(new_node)
            node.edges.append(new_edge)

        return value


    def backup(self, node, value):
        if node.isRoot():            
            return
        
        for idx in range(sum(node._parent.legal_moves)):
            if node is node._parent.children[idx]:
                break
        
        edge = node._parent.edges[idx]
        edge.status["N"] += 1
        edge.status["W"] += value
        edge.status["Q"] = edge.status["W"] / edge.status["N"]
        
        self.backup(node._parent, value)
            

    def tree_search(self):        
        node = self.root_node
        while True:
            if node.isLeaf():                
                break
            node = self._select(node)

        value = self._expand_and_evaluate(node)        
        self.backup(node, value)
        self.search_count += 1
        
    def get_probs(self, tau):
        N_values = []
        Q_values = []
        P_values = []
        N_total = 0
        for edge in self.root_node.edges:
            N_values.append(edge.status["N"])
            Q_values.append(edge.status["Q"])
            P_values.append(edge.status["P"])
            N_total += edge.status["N"]
        
        
        legal_probs = list((np.array(N_values) / N_total) ** 1)                    
        probs = self.root_node.legal_moves.copy()
        for idx in range(4):
            if probs[idx]:
                p = legal_probs.pop(0)
                probs[idx] *= p  

        # if tau != 0:            
        #     legal_probs = list((np.array(N_values) / N_total) ** tau)
                    
        #     probs = self.root_node.legal_moves.copy()
        #     for idx in range(4):
        #         if probs[idx]:
        #             p = legal_probs.pop(0)
        #             probs[idx] *= p                    
        # else:
        #     legal_moves = self.root_node.legal_moves.copy()
        #     # max_n = max(N_values)
        #     # for idx in range(4):
        #     #     if probs[idx]:
        #     #         n_val = N_values.pop()
        #     #         p = int(n_val == max_n)
        #     #         probs[idx] = p
        #     for _ in range(4):
        #         try:
        #             idx = legal_moves.index(0)
        #         except ValueError:
        #             break
        #         N_values.insert(idx, legal_moves.pop(idx))
        #         legal_moves.insert(0,1)
        #     probs = [0] * 4
        #     probs[np.argmax(N_values)] = 1
        assert abs(sum(probs)) <= 1 + 1e+5
        return probs, [Q_values, N_values, P_values]
