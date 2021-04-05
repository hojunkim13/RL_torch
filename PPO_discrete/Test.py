from Agent import Agent
import numpy as np
from PongWrapper import Environment
import time

env_name = 'Pong-v0'
env = Environment()
n_episode = 1
render = True

state_dim = (1,84,84)
action_dim = 3
gamma = 0.99
lmbda = 0.95
lr = 1e-4
time_step = 1024
K_epochs = 10
batch_size = 64
epsilon = 0.2
path = 'model/' + env_name

agent = Agent(state_dim, action_dim, lr, gamma, lmbda, epsilon, time_step, K_epochs, batch_size)
agent.load(path)


if __name__ == "__main__":
    score_list = []
    mas_list = []
    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        while not done:
            if render:
                env.render()
                time.sleep(0.01)
            action, log_prob = agent.get_action(state)
            state_, reward, done, _ = env.step(action + 1, False)
            score += reward
            state = state_
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

