from Agent import Agent
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Env import Game2048_wrapper
from _2048 import Game2048
import pygame

p1 = os.path.join("save", '2048_.score')
p2 = os.path.join("save", '2048_.%d.state')        
screen = pygame.display.set_mode((Game2048.WIDTH, Game2048.HEIGHT))
pygame.init()
pygame.display.set_caption("2048!")
pygame.display.set_icon(Game2048.icon(32))
env_name = '2048'
env = Game2048_wrapper(screen, p1, p2)


state_dim = (1,4,4)
action_dim = 4

n_episode = 250
load = False
save_freq = 10
gamma = 0.99
lmbda = 0.95
alpha = 5e-4
beta = 5e-4
time_step = 20
K_epochs = 3
epsilon = 0.1
agent = Agent(state_dim, action_dim, alpha, beta, gamma, lmbda, epsilon, time_step, K_epochs)
agent.actor.eval()
agent.critic.eval()
agent.load(env_name)

if __name__ == "__main__":
    score_list = []
    mas_list = []
    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset(True)
        while not done:           
            env.draw()
            action, prob = agent.get_action(state)
            state_, reward, done = env.step(action)
            score += reward
            state = state_
        #done
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')
    env.close()

