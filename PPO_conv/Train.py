from Agent import Agent
import gym
import numpy as np

env_name = 'LunarLander-v2'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

n_episode = 1000
load = True
render = False
save_freq = 10
gamma = 0.98
lmbda = 0.95
alpha = 5e-4
beta = 5e-4
time_step = 20
K_epochs = 3
epsilon = 0.1
path = 'model/' + env_name

agent = Agent(state_dim, action_dim, alpha, beta, gamma, lmbda, epsilon, time_step, K_epochs)
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
                if render:
                    env.render()
                action, prob = agent.get_action(state)
                state_, reward, done, _ = env.step(action)
                score += reward
                agent.store(state, action, reward, state_, done, prob)
                state = state_
                if done:
                    break
            agent.learn()
        #done
        if (e+1) % save_freq == 0:
            agent.save(path)
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

