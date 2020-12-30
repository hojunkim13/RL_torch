import numpy as np

class ReplayBuffer:
    def __init__(self, rule):
        self.batch_size = rule.batch_size
        self.maxlen = rule.maxlen
        self.S  = np.zeros((rule.maxlen, rule.state_dim), 'float') 
        self.A  = np.zeros((rule.maxlen, rule.action_dim), 'float') 
        self.R  = np.zeros((rule.maxlen, 1), 'float') 
        self.S_ = np.zeros((rule.maxlen, rule.state_dim), 'float')
        self.D  = np.zeros((rule.maxlen, 1), 'bool') 
        self.mem_counter = 0

    def store(self, s, a, r, s_, d):
        idx = self.mem_counter % self.maxlen
        
        self.S[idx] = s
        self.A[idx] = a
        self.R[idx] = r
        self.S_[idx] = s_
        self.D[idx] = d
        self.mem_counter += 1

    def get_samples(self):
        length = min(self.mem_counter, self.maxlen)
        indices = np.random.choice(np.arange(length), size = self.batch_size)
        S  = self.S[indices]
        A  = self.A[indices]
        R  = self.R[indices]
        S_ = self.S_[indices]
        D  = self.D[indices]
        return S, A, R, S_, D
