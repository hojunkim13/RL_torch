import numpy as np
from collections import deque

class PrioritizedExpericeReplay:
  
    def __init__(self, capacity, n_state, n_action, td_error_epsilon = 1e-4):
        """
        capacity : capacity of memory
        memory : variable for saving td-error
        index : index for saving location
        TD_ERROR_EPSILON : Prevent zero possibility
        """
        self.TD_ERROR_EPSILON = td_error_epsilon
        self.capacity = capacity
        self.n_action = n_action
        self.n_state = n_state
        if type(n_state) is tuple:
            self.S = np.zeros((capacity,) + n_state, dtype = float)
            self.S_ = np.zeros((capacity,) + n_state, dtype = float)
        else:
            self.S = np.zeros((capacity,n_state), dtype = float)
            self.S_ = np.zeros((capacity,n_state), dtype = float)            
        self.A = np.zeros((capacity,1), dtype = np.uint8)
        self.R = np.zeros((capacity,1), dtype = float)
        self.D = np.zeros((capacity,1), dtype = bool)
        self.T = np.zeros((capacity, 1), dtype = np.float)
        self.mem_cntr = 0          
        
    def __len__(self):
        return self.mem_cntr
    
    def stackMemory(self, s, a, r, s_, d, td_error):
        idx = self.mem_cntr % self.mem_max
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.T[idx] = td_error
        self.mem_cntr += 1


    def getSample(self, n):
        
        indice = self.getIndice(n)
        S = self.S[indice]
        A = self.A[indice]
        R = self.R[indice]
        S_ = self.S_[indice]
        D = self.D[indice]
        return (S, A, R, S_, D)

    def getIndice(self, n):
        max_indice = min(self.capacity, self.mem_cntr)
        possiblites = np.abs(self.T) ** self.alpha + self.TD_ERROR_EPSILON
        possiblites /= np.sum(possiblites)
        indice = np.random.choice(range(max_indice), n, p = possiblites)
        return indice


