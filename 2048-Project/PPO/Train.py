import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PPO.Agent import Agent
import numpy as np
from Environment.DosEnv import _2048
from Environment.Utils import *

env_name = "2048"
env = _2048()
state_dim = (20,4,4)
action_dim = 4


n_episode = 50000
load = True
save_freq = 100
gamma = 0.99
lmbda = 0.95
alpha = 1e-4
beta = 1e-4
buffer_size = 1024
batch_size = 128
k_epochs = 10
epsilon = 0.2
agent = Agent(state_dim, action_dim, alpha, beta, gamma, lmbda, epsilon, buffer_size, batch_size, k_epochs)


if load:
    agent.load(env_name)

if __name__ == "__main__":
    score_list = []
    mas_list = []
    for e in range(n_episode):
        done = False
        score = 0
        grid = env.reset()
        state = preprocessing(grid)
        while not done:            
            action, log_prob = agent.get_action(state)
            grid, reward, done, info = env.step(action)
            state_ = preprocessing(grid)
            score += reward
            agent.store(state, action, reward, state_, done, log_prob)                
            agent.learn()
            state = state_
        #done
        if (e+1) % save_freq == 0:
            agent.save(env_name)
        score_list.append(score)
        average_score = np.mean(score_list[-1000:])
        mas_list.append(average_score)
        print(f"Episode : {e+1} / {n_episode}, Score : {score:.0f}, Average: {average_score:.1f}, Max : {info}")

