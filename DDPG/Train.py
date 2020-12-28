from Agent import Agent
import gym
from Rule import Rule
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    rule = Rule()
    agent = Agent(rule)
    env = gym.make(rule.env_name)
    score_list = []
    average_score_list = []
    for e in range(rule.epochs):
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.get_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            agent.replaybuffer.store(state, action, reward, state_, done)
            agent.learn()
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        print(f'[{e+1}/{rule.epochs}] [Score: {score:.0f}] [Average Score: {average_score:.0f}]')

    plt.plot(np.arange(rule.epochs), average_score_list)
    plt.xlabel('Epochs')
    plt.ylabel('Moving Average Score')
    plt.title(rule.env_name)
    plt.show()
    
    