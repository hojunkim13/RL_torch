from Agent import Agent
import gym
import numpy as np
from Environment import Environment




env_name = 'CarRacing-v0'


n_episode = 1000
frame_skip = 4
frame_stack = 4
env = Environment(env_name, frame_skip, frame_stack)

state_dim = (4, 96, 96)
action_dim = 5
load = True
render = True
save_freq = 10
gamma = 0.99
lmbda = 0.95
lr = 1e-4
time_step = 100
K_epochs = 15
epsilon = 0.1
path = 'model/' + env_name

agent = Agent(state_dim, action_dim, lr, gamma, lmbda, epsilon, time_step, K_epochs)


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
            for _ in range(time_step):
                action, prob = agent.get_action(state)
                state_, reward, done, _ = env.step(action, render)
                score += reward
                agent.store(state, action, reward, state_, done, prob)
                state = state_
                if done:
                    break
            agent.learn()
        #done
        env.env.close()
        if (e+1) % save_freq == 0:
            agent.save(path)
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

