import gym
from Agent import ACAgent
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = 'MountainCar-v0'
alpha = 1e-4
beta = 5e-4
gamma = 0.99
EPOCHS = 500
PATH =  'model/'

load = False
SAVE_FREQ = 10
render = True

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ACAgent(obs_dim, action_dim, alpha, beta, gamma)
    reward_adjustment = True if ENV_NAME == 'MountainCar-v0' else False
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
            if reward_adjustment:
                reward = 1 / (0.5 - obs_[0]) * abs(obs_[1])
                reward = np.clip(reward, -1., 1.)
                if done and obs_[0] > 0.45:
                    reward += 10
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