import numpy as np
from collections import deque

class PrioritizedExpericeReplay:
  
    def __init__(self, TD_ERROR_EPSILON = 1e-4, CAPACITY = int(1e+5)):
        """
        capacity : capacity of memory
        memory : variable for saving td-error
        index : index for saving location
        TD_ERROR_EPSILON : Prevent zero possibility
        """        
        self.memory = deque(maxlen = CAPACITY)
        self.TD_ERROR_EPSILON = TD_ERROR_EPSILON
        self.index = 0
        
    def __len__(self):
        return len(self.memory)

    def push(self, td_error):
        """[summary] Save TD-error in memory

        Args:
            td_error ([type]): [description]
        """                
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        index = self.index % self.capacity
        self.memory.append(td_error)

    def get_prioritized_indice(self, batch_size):
    
        sum_absolute_td_error = np.sum(np.abs(self.memory))
        sum_absolute_td_error += self.TD_ERROR_EPSILON * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)
        
        indice = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx]) + self.TD_ERROR_EPSILON)
                idx += 1