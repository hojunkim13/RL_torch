import numpy as np
from Environment.DosEnv import _2048
from Environment.Utils import *
import time
from threading import Thread


n_sim = 200
n_episode = 5

env_name = "2048"
env = _2048()


class MCTS:
    def __init__(self, n_sim):
        self.n_sim = n_sim
        self.values = np.zeros(4)
        self.visits = np.zeros(4)

    def setRootGrid(self, grid):
        self.root_grid = grid
        self.values = np.zeros(4)
        
    def slmulation(self):        
        for first_action in range(4):
            start_grid = move_grid(self.root_grid, first_action)
            if np.array_equal(start_grid, self.root_grid):                            
                continue
            for sim in range(self.n_sim // 4):
                value = 0
                grid = move_grid(self.root_grid, first_action)
                while not isEnd(grid):
                    action = np.random.randint(0,4)
                    grid = move_grid(grid, action)
                    value += 1                                
                self.values[first_action] += value            
        outputs = self.values
        return np.argmax(outputs)        

    def simulThread(self, first_action):
        start_grid = move_grid(self.root_grid, first_action)
        if np.array_equal(start_grid, self.root_grid):
            return
        for _ in range(self.n_sim // 4):
            grid = move_grid(self.root_grid, first_action)
            value = 0
            while not isEnd(grid):
                action = np.random.randint(0,4)
                grid = move_grid(grid, action)
                value += 1
            
            self.values[first_action] += value        
        
        

    def simulWithThread(self):
        threads = []
        for first_action in range(4):
            worker = Thread(target = self.simulThread, args = (first_action,))
            worker.start()
            threads.append(worker)
        [worker.join() for worker in threads]        
        return np.argmax(self.values)
       

def main():
    mcts = MCTS(n_sim)
    score_list = []
    for e in range(n_episode):
        start_time = time.time()
        done = False
        grid = env.reset()
        while not done:
            mcts.setRootGrid(grid)
            #action = mcts.slmulation()
            action = mcts.simulWithThread()
            grid, _, done, info = env.step(action, False)
        score_list.append(info)
        average_score = np.mean(score_list[-100:])        
        spending_time = time.time() - start_time
        print(f"Episode : {e+1} / {n_episode}, Max Tile : {info}, Average: {average_score:.1f}")
        print(f"SPENDING TIME : {spending_time:.1f} Sec")
        #time.sleep(3)

if __name__ == "__main__":
    main()

