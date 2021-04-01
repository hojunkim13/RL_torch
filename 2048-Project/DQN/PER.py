from SumTree import SumTree
import numpy as np

class PrioritizedExperienceReplay:
    epslion = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    
    def _calc_priority(self, error):
        priority = (np.abs(error) + self.epslion) ** self.alpha
        return priority

    def add(self, transition, error):
        priority = self._calc_priority(error)
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        indice = []
        priorities = []
        datas = []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            data = 0
            while data == 0:
                s = np.random.uniform(a,b)
                (idx, priority, data) = self.tree.get(s)
            indice.append(idx)
            priorities.append(np.clip(priority, 1e-6, 1-1e-6))
            datas.append(data)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        datas = np.array(datas, dtype= object)

        return datas, indice, is_weight

    def update(self, idx, error):
        p = self._calc_priority(error)
        self.tree.update(idx, p)