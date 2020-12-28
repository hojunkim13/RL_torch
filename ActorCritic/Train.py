import gym
from Agent import ACAgent
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = 'CartPole-v1'
alpha = 1e-4
beta = 5e-4
gamma = 0.99
EPOCHS = 300
SAVE_FREQ = 5000
PATH =  'model/'

load = False
render = False

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ACAgent(obs_dim, action_dim, alpha, beta, gamma)
    if load:
        agent.load(PATH)
    score_list = []

    for e in range(EPOCHS):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            if render:
                env.render()
            action = agent.get_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward
            agent.train(obs, action, reward, obs_, done)
            obs = obs_
        # Episode is done.
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        print(f'[{e+1}/{EPOCHS}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')
        if e % SAVE_FREQ == (SAVE_FREQ-1):
            agent.save(PATH)
    
    env.close()
    plt.plot(range(EPOCHS), score_list)