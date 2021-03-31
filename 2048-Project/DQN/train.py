from _2048 import Game2048
import os, sys
path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(path)
from Env import Game2048_wrapper, move
import numpy as np
import pygame
from Agent import Agent
import matplotlib.pyplot as plt

env_name = "2048"
load = False
n_state = (16,4,4)
n_action = 4
learing_rate = 1e-4
gamma = 0.9
replay_memory_buffer_size = 100000
epsilon_decay = 0.95
epsilon_decay_step = 2500
epsilon_min = 0.1
batch_size = 256
tau = 1e-3

n_episode = 10000


p1 = os.path.join("save", '2048_.score')
p2 = os.path.join("save", '2048_.%d.state')        
screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
pygame.init()
pygame.display.set_caption("2048!")
pygame.display.set_icon(Game2048.icon(32))
  
if __name__ == "__main__":
    env = Game2048_wrapper(screen, p1, p2)
    agent = Agent(n_state, n_action, learing_rate, gamma, replay_memory_buffer_size,
                epsilon_decay,epsilon_min,epsilon_decay_step, batch_size, tau)

    agent.load(env_name)
    agent.epsilon = 0.1
    agent.optimizer.param_groups[0]["lr"] = 1e-5
    
    scores = []
    env.draw()
    for episode in range(n_episode):
        state = env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            action = agent.getAction(state)
            state_, reward, done = env.step(action)
            score += reward
            agent.storeTransition(state, action, reward, state_, done)
            agent.learn()
            agent.softUpdate()
            state = state_ 
            step += 1
        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        
        if (episode + 1) % 100 == 0:
            agent.save(env_name)
        lr = agent.optimizer.param_groups[0]["lr"]
        print(f"Episode : {episode+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon:.3f}, Memory : {agent.memory.tree.n_entries}, LearningRate : {lr:1.2e}")

    pygame.quit()
    env.close()
        
            
