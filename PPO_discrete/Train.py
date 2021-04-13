from Agent import Agent
import numpy as np
from AtariWrapper import Environment

#env_name = 'Pong-v0'
env_name = 'Breakout-v0'
env = Environment(env_name, record = False)
state_dim = (1,80,80)
action_dim = 3
n_episode = 50000
load = True
render = False
save_freq = 100
gamma = 0.99
lmbda = 0.95
lr = 1e-5
buffer_size = 2000
K_epochs = 10
batch_size = 128
epsilon = 0.20
path = 'model/' + env_name

agent = Agent(state_dim, action_dim, lr, gamma, lmbda, epsilon, buffer_size, K_epochs, batch_size)
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
            #env.imgshow(state_)
            score += reward
            agent.store(state, action, reward, state_, done, log_prob)
            state = state_
            agent.learn()
        #done
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)

        if (e+1) % save_freq == 0:
            agent.save(path)
        if (e+1) % 10 == 0:
            print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')


