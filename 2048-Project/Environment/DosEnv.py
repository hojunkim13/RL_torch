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

    def step(self, action, render):
        action = int(action)
        grid, changed = self.action_space[action](self.grid)
        if changed:
            logic.add_new_tile(grid)
        
        game_state = logic.get_current_state(grid)        
        if game_state in ("WON", "LOST"):
            done = True
        else:
            done = False
            
        reward = self._calcReward(grid, changed, done)
        self.score += reward
        self.grid = grid
        if render:
            self.render()
            act_name = {0:"LEFT", 1:"UP", 2:"RIGHT", 3:"DOWN"}[action]
            t_log = time.time() - self.time_log
            self.time_log = time.time()
            print(f"Move Direction : {act_name}, Thinking time: {t_log:.2f} sec")
        return grid, reward, done, int(np.max(grid))


    def _calcReward(self, grid, changed, done):
        if done:
            return -100
        elif not changed:
            return -1
        grid = np.array(grid).reshape(-1)
        reward1 = np.log2(grid.max()) * 2
        reward2 = (len(grid) - np.count_nonzero(grid))
        return reward1 + reward2
    
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
    
    def close(self):
        pass