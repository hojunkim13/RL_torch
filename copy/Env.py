import os, pygame
import numpy as np
from _2048.game import Game2048
from _2048.manager import GameManager


class Game2048_wrapper(GameManager):
    def __init__(self, screen, p1, p2):
        super().__init__(Game2048, screen, p1, p2)    
        self.actionSpace =  [
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}),   # UP
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}), # RIGHT
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}), # DOWN
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT}) # LEFT
            ]

        data_dir = "save"
        os.makedirs(data_dir, exist_ok=True)

    def preprocessing(self):
        state = self.game.grid
        state = np.clip(state, 1, None)
        state = np.log2(state)
        return state.reshape(16)

    def free_cells(self):
        grid = self.game.grid
        return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]

    def reset(self, test_mode = False):
        self.new_game()
        if not test_mode:
            self.game.ANIMATION_FRAMES = 1
            self.game.WIN_TILE = 999999
        state = self.game.grid
        state = self.preprocessing()
        pygame.init()
        pygame.display.set_caption("2048!")
        pygame.display.set_icon(Game2048.icon(32))
        return state
    
    def step(self, action):
        old_grid = self.game.grid
        old_score = self.game.score
        event = self.actionSpace[action]
        self.dispatch(event)
        state = self.preprocessing()
        if self.game.grid == old_grid:
            reward = -1
        else:
            #reward = self.calcReward(state)
            reward = self.game.score - old_score
        return state, reward
            
    def calcReward(self, state):
        left = np.clip(len(self.free_cells()), 1, None)
        vertex_value = np.clip(state[[0,3,12,15]].sum(), 1, None)

        reward1 = np.log(left)
        reward2 = -np.log2(np.std(state))
        reward3 = np.log(vertex_value)
        reward = reward1 + reward2 + reward3
        return reward