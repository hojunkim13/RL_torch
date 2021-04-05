from Agent import Agent
import numpy as np
from PongWrapper import Environment

env_name = 'Pong-v0'
env = Environment()
state_dim = (1,84,84)
action_dim = 3
n_episode = 100000
load = True
render = False
save_freq = 100
gamma = 0.99
lmbda = 0.95
lr = 1e-4
time_step = 1024
K_epochs = 10
batch_size = 64
epsilon = 0.2
path = 'model/' + env_name

agent = Agent(state_dim, action_dim, lr, gamma, lmbda, epsilon, time_step, K_epochs, batch_size)
if load:
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
            action, log_prob = agent.get_action(state)
            state_, reward, done, _ = env.step(action + 1, False)
            score += reward
            agent.store(state, action, reward, state_, done, log_prob)
            state = state_
            agent.learn()
        #done
        if (e+1) % save_freq == 0:
            agent.save(path)
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

