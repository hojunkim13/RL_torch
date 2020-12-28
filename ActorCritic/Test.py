import gym
from Agent import ACAgent
import numpy as np

ENV_NAME = 'CartPole-v1'
PATH =  'model/'
alpha = 1e-4
beta = 5e-4
gamma = 0.99
EPOCHS = 10

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    agent = ACAgent(4, 2, alpha, beta, gamma)
    agent.load(PATH)
    score_list = []

    for e in range(EPOCHS):
        done = False
        score = 0
        obs = env.reset()
        while not done:

            env.render()
            action = agent.get_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward
            obs = obs_
        # Episode is done.
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        print(f'[{e+1}/{EPOCHS}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')
    env.close()