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
        self.state = []
        self.value_weights = np.array([[2,1,1,2],
                                        [1,-1,-1,1],
                                        [1,-1,-1,1],
                                        [2,1,1,2],
                                        ])
        data_dir = "./2048-Project/save"
        os.makedirs(data_dir, exist_ok=True)

    def getState(self):
        state = np.array(self.game.grid)
        state = np.clip(state, 1, None)
        state = np.log2(state) / 10
        return state.reshape(1,4,4)

    def getFreeCells(self):
        grid = self.game.grid
        return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]

    def reset(self, test_mode = False):
        self.new_game()
        if not test_mode:
            self.game.ANIMATION_FRAMES = 1
            self.game.WIN_TILE = 2048
        state = self.getState()
        pygame.init()
        pygame.display.set_caption("2048!")
        pygame.display.set_icon(Game2048.icon(32))
        return state
    
    def step(self, action):
        event = self.actionSpace[action]
        self.dispatch(event)
        state = self.getState()
        #reward = self.calcReward(state)
        try:            
            if self.game.score < self.old_score:
                self.old_score = 0
            reward = self.game.score - self.old_score
            if reward == 0:
                reward = -5
        except AttributeError:
            reward = 0
        self.old_score = self.game.score

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()                
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dispatch(event)
        done = self.game.won or self.game.lost
        return state, reward, done
            
    def calcReward(self, state):
        if np.array_equal(self.state, state):
            return -5
        self.state = state
        number_value = np.clip((self.value_weights * state).sum(), 1e-4, None)
        reward1 = len(self.getFreeCells())
        reward2 = -np.log(np.std(state))
        reward3 = np.log(number_value)
        reward = reward1 + reward2 + reward3
        return np.clip(reward, 0, 5)