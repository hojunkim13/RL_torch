from _2048.manager import GameManager
from _2048.game import Game2048
import pygame
import os
import numpy as np
import random

CELLS = [
  [(r, c) for r in range(4) for c in range(4)], # LEFT
  [(r, c) for c in range(4) for r in range(4)], # UP
  [(r, c) for r in range(4) for c in range(4 - 1, -1, -1)], # RIGHT
  [(r, c) for c in range(4) for r in range(4 - 1, -1, -1)], # DOWN
]

GET_DELTAS = [
  lambda r, c: ((r, i) for i in range(c + 1, 4)), # LEFT
  lambda r, c: ((i, c) for i in range(r + 1, 4)), # UP
  lambda r, c: ((r, i) for i in range(c - 1, -1, -1)), # RIGHT
  lambda r, c: ((i, c) for i in range(r - 1, -1, -1)), # DOWN
]

def free_cells(grid):
  return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]

def move(grid, action):
    moved, sum = 0, 0
    for row, column in CELLS[action]:
        for dr, dc in GET_DELTAS[action](row, column):
            # If the current tile is blank, but the candidate has value:
            if not grid[row][column] and grid[dr][dc]:
                # Move the candidate to the current tile.
                grid[row][column], grid[dr][dc] = grid[dr][dc], 0
                moved += 1
            if grid[dr][dc]:
            # If the candidate can merge with the current tile:
                if grid[row][column] == grid[dr][dc]:
                    grid[row][column] *= 2
                    grid[dr][dc] = 0
                    sum += grid[row][column]
                    moved += 1
            # When hitting a tile we stop trying.
                break
    if moved:
        grid = spawn_new(grid)
    return grid, moved, sum

def spawn_new(grid):
    """Spawn some new tiles."""
    free = free_cells(grid)
    x, y = random.choice(free)
    grid[y][x] = random.randint(0, 10) and 2 or 4
    return grid


def getState(grid = None):
    if grid is None:
        grid = self.game.grid
    state = np.array(grid).reshape(-1)
    state = np.clip(state, 1, None)
    state = np.log2(state)
    state = np.eye(16)[state.astype(int)]
    state = state.reshape(4,4,16)
    return np.transpose(state, (2,0,1))



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
        data_dir = "./2048-Project/save"
        os.makedirs(data_dir, exist_ok=True)

    def state2grid(self, state):
        grid = np.transpose(state, (1,2,0))
        grid = np.argmax(grid, axis = -1)
        grid = 2 ** (grid)
        grid = np.where(grid==1, 0, grid).reshape(4,4)
        return grid

    def getState(self, grid = None):
        if grid is None:
            grid = self.game.grid
        state = np.array(grid).reshape(-1)
        state = np.clip(state, 1, None)
        state = np.log2(state)
        state = np.eye(16)[state.astype(int)]
        state = state.reshape(4,4,16)
        return np.transpose(state, (2,0,1))

    def getFreeCells(self):
        grid = self.game.grid
        return [(x, y) for x in range(4) for y in range(4) if not grid[y][x]]

    def reset(self, test_mode = False):
        self.new_game()
        if not test_mode:
            self.game.ANIMATION_FRAMES = 1
            self.game.WIN_TILE = 2048
        state = self.getState()
        self.old_score = 0
        return state
    
    def step(self, action):
        event = self.actionSpace[action]
        self.dispatch(event)
        state = self.getState()
        reward = self.calcReward()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dispatch(event)
        done = self.game.won or self.game.lost
        if done:
            reward = -1
        self.old_score = self.game.score
        return state, reward, done
            
    def calcReward(self):
        if np.array_equal(self.old_score, self.game.score):
            return 0
        
        #reward = self.game.score - self.old_score
        grid = np.array(self.game.grid).reshape(-1)
        reward1 = np.log2(grid.max()) * 2
        reward2 = (len(grid) - np.count_nonzero(grid))
        return reward1 + reward2
