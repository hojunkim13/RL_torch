from _2048.manager import GameManager
from _2048.game import Game2048
import pygame
import os
import numpy as np



class Game2048_wrapper(GameManager):
    def __init__(self, screen, p1, p2):
        super().__init__(Game2048, screen, p1, p2)
        pygame.init()
        pygame.display.set_caption("2048!")
        pygame.display.set_icon(Game2048.icon(32))
        self.actionSpace =  [
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT}), # LEFT
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}),   # UP
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}), # RIGHT
            pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}), # DOWN
            ]
        data_dir = "./data/game/"
        os.makedirs(data_dir, exist_ok=True)


    def reset(self, test_mode = True):
        self.new_game()
        if not test_mode:
            self.game.ANIMATION_FRAMES = 1
            self.game.WIN_TILE = 999999
        else:
            self.game.ANIMATION_FRAMES = 10
            self.game.WIN_TILE = 999999
        return self.game.grid
    
    def step(self, action):
        old_score = self.game.score
        event = self.actionSpace[action]
        self.dispatch(event)
        state = self.game.grid        
        reward = self.calcReward(old_score)            
        done = self.game.won or self.game.lost
        info = np.max(state)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dispatch(event)        
        return state, reward, done, info
            
    def calcReward(self, old_score):
        if self.game.lost or self.game.won:
            return -100
        elif self.game.score == old_score:
            return -1
        grid = np.array(self.game.grid).reshape(-1)
        reward1 = np.log2(grid.max()) * 2
        reward2 = (len(grid) - np.count_nonzero(grid))
        return reward1 + reward2
