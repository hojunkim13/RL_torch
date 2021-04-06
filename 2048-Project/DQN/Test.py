import numpy as np
import os
import pygame
from Agent import Agent
from Utils import *


env_name = "2048"
load = False
n_state = (20,4,4)
n_action = 4
learing_rate = 1e-3
gamma = 0.99
replay_memory_buffer_size = 10000
epsilon_decay = 0.999
epsilon_min = 0.1
batch_size = 256

from _2048 import Game2048
from Environment.PrettyEnv import Game2048_wrapper
import pygame
p1 = os.path.join("data/game", '2048_.score')
p2 = os.path.join("data/game", '2048_.%d.state')        
screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
pygame.init()
pygame.display.set_caption("2048!")
pygame.display.set_icon(Game2048.icon(32))
env = Game2048_wrapper(screen, p1, p2)
env.draw()

  
if __name__ == "__main__":
    
    agent = Agent(n_state, n_action, learing_rate, gamma, replay_memory_buffer_size,
                epsilon_decay,epsilon_min, batch_size, gamma, gamma)
    agent.net.eval()
    agent.net_.eval()
    agent.load(env_name,)
    n_episode = 10
    scores = []
    env.draw()
    for episode in range(n_episode):
        grid = env.reset(test_mode=True)
        state = preprocessing(grid)
        done = False
        score = 0
        while not done:
            action, _ = agent.getAction(state, True)
            grid, reward, done = env.step(action)
            state_ = preprocessing(grid)
            score += reward
            state = state_
            
        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        print(f"Episode : {episode+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")
    pygame.quit()
    env.close()
