import logic
import numpy as np
import os

class _2048:
    def __init__(self):
        self.action_space = [
                            logic.move_left,
                            logic.move_up,
                            logic.move_right,
                            logic.move_down,                            
                            ]

    def step(self, action, grid = None):
        if grid is None:
            grid = self.grid
        grid, changed = self.action_space[action](grid)
        game_state = logic.get_current_state(grid)
        if game_state in ("WON", "LOST"):
            done = True
        else:
            done = False
            if changed:
                logic.add_new_tile(grid)
        reward = self._calcReward(grid, changed, done)
        self.score += reward
        self.grid = grid
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
        return self.grid

    def render(self):
        os.system("cls")
        print(self.grid[0])
        print(self.grid[1])
        print(self.grid[2])
        print(self.grid[3])
    
    def close(self):
        pass