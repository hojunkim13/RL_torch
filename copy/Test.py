import numpy as np
import os, time
from _2048 import Game2048
import pygame
from Env import Game2048_wrapper
from Agent import Agent

env_name = "2048"
load = False
n_state = 16
n_action = 4
learing_rate = 1e-3
gamma = 0.99
replay_memory_buffer_size = 10000
epsilon_decay = 0.999
epsilon_min = 0.1
network_sync_freq = 50
batch_size = 256


p1 = os.path.join("save", '2048_.score')
p2 = os.path.join("save", '2048_.%d.state')        
screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
pygame.init()
pygame.display.set_caption("2048!")
pygame.display.set_icon(Game2048.icon(32))
  
if __name__ == "__main__":
    env = Game2048_wrapper(screen, p1, p2)
    agent = Agent(n_state, n_action, learing_rate, gamma, replay_memory_buffer_size,
                epsilon_decay,epsilon_min, batch_size)    
    agent.net.eval()
    agent.net_.eval()
    agent.load(env_name)
    n_episode = 1
    scores = []
    score_old = 0
    for episode in range(n_episode):
        state = env.reset(True)
        done = False
        score = 0
        while not done:
            action = agent.getAction(state, True)
            print(action)
            state_, reward = env.step(action)                
            done = env.game.lost or env.game.won
            score += reward
            state = state_
            env.draw()            
        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        print(f"Episode : {episode+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")
    pygame.quit()
    env.close()
