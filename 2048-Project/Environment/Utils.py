import random
import numpy as np

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
        grid = _spawn_new(grid)
    return grid, moved, sum


def _spawn_new(grid):
    """Spawn some new tiles."""
    free = free_cells(grid)
    x, y = random.choice(free)
    grid[y][x] = random.randint(0, 10) and 2 or 4
    return grid

def state2grid(state):
    grid = np.transpose(state, (1,2,0))
    grid = np.argmax(grid, axis = -1)
    grid = 2 ** (grid)
    grid = np.where(grid==1, 0, grid).reshape(4,4)
    return grid

def preprocessing(grid):
    legal_action_plane = np.ones((4,4,4))
    for action in range(4):
        _, moved, _ = move(grid, action)
        if not moved:
            legal_action_plane[:,:,action] = 0

    state = np.array(grid).reshape(-1)
    state = np.clip(state, 1, None)
    state = np.log2(state)
    state = np.eye(16)[state.astype(int)]
    state = state.reshape(4,4,16)
    state = np.concatenate((legal_action_plane, state), axis = -1)
    return np.transpose(state, (2,0,1))