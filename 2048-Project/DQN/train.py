import numpy as np
import os, sys
from _2048 import Game2048
import pygame
from Agent import Agent
from Env import Game2048_wrapper

env_name = "2048"
load = False
n_state = (1,4,4)
n_action = 4
learing_rate = 1e-4
gamma = 0.99
replay_memory_buffer_size = 10000
epsilon_decay = 0.999
epsilon_min = 0.05
batch_size = 64
tau = 1e-3



p1 = os.path.join("save", '2048_.score')
p2 = os.path.join("save", '2048_.%d.state')        
screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
pygame.init()
pygame.display.set_caption("2048!")
pygame.display.set_icon(Game2048.icon(32))
  
if __name__ == "__main__":
    env = Game2048_wrapper(screen, p1, p2)
    agent = Agent(n_state, n_action, learing_rate, gamma, replay_memory_buffer_size,
                epsilon_decay,epsilon_min, batch_size,tau)

    agent.load(env_name)
    n_episode = 200
    scores = []
    for episode in range(n_episode):
        state = env.reset()
        done = False
        score = 0
        env.draw()
        while not done:
            action = agent.getAction(state)
            state_, reward, done = env.step(action)
            score += reward
            agent.storeTransition((state, action, reward, state_, done))
            agent.learn()
            agent.softUpdate()
            state = state_ 
        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        
        if movingAverageScore <= score:
            agent.save(env_name)
        
        print(f"Episode : {episode+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")
    pygame.quit()
    env.close()
        
            
