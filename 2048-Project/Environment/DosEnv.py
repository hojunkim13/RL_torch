from Environment import logic
import numpy as np
import os
import time

class _2048:
    def __init__(self):
        self.action_space = [
                            logic.move_left,
                            logic.move_up,
                            logic.move_right,
                            logic.move_down,                            
                            ]
        self.goal = 999999


    def step(self, action):
        action = int(action)
        grid, changed = self.action_space[action](self.grid)
        if changed:
            logic.add_new_tile(grid)
        
        game_state = logic.get_current_state(grid, self.goal)        
        if game_state in ("WON", "LOST"):
            done = True
        else:
            done = False
            
        reward = self._calcReward(grid, changed, done)
        self.score += reward
        self.grid = grid

        return grid, reward, done, np.max(grid)

    def _calcReward(self, grid, changed, done):
        if (not changed) or done:
            return 0        
        return np.sum(grid) - np.sum(self.grid)
    
    def reset(self):
        self.grid = logic.start_game()
        self.score = 0
        self.time_log = time.time()
        return self.grid

    def render(self):
        os.system("cls")
        print(self.grid[0])
        print(self.grid[1])
        print(self.grid[2])
        print(self.grid[3])
        spending_time = time.time() - self.time_log
        self.time_log = time.time()
        print(spending_time)

    def close(self):
        pass