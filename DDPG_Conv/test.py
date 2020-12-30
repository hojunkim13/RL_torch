from Agent import Agent
import gym
from Rule import Rule
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    n_episode = 10
    rule = Rule()
    rule.load = True
    agent = Agent(rule)
    env = gym.make(rule.env_name)
    score_list = []
    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        while not done:
            env.render()
            action = agent.get_action(state, eval = True)
            state_, reward, done, _ = env.step(action)
            score += reward
            state = state_
        score_list.append(score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.0f}]')
    plt.plot(np.arange(n_episode), score_list)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(rule.env_name)
    plt.show()
