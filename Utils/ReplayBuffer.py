import numpy as np

class ReplayBuffer:
    def __init__(self, mem_max, n_state, n_action):
        self.mem_max = mem_max
        self.n_action = n_action
        self.n_state = n_state
        if type(n_state) is tuple:
            self.S = np.zeros((mem_max,) + n_state, dtype = float)
            self.S_ = np.zeros((mem_max,) + n_state, dtype = float)
        else:
            self.S = np.zeros((mem_max,n_state), dtype = float)
            self.S_ = np.zeros((mem_max,n_state), dtype = float)
        self.A = np.zeros((mem_max,1), dtype = "uint8")
        self.R = np.zeros((mem_max,1), dtype = float)
        self.D = np.zeros((mem_max,1), dtype = bool)
        self.mem_cntr = 0  

    def stackMemory(self, s, a, r, s_, d):
        idx = self.mem_cntr % self.mem_max    
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.mem_cntr += 1


    def getSample(self, n = 1):
        max_indice = min(self.mem_cntr, self.mem_max)
        indice = np.random.randint(0, max_indice, size = n)
        S = self.S[indice]
        A = self.A[indice]
        R = self.R[indice]
        S_ = self.S_[indice]
        D = self.D[indice]
        return (S, A, R, S_, D)