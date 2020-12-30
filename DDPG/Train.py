from Agent import Agent
import gym
from Rule import Rule
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    rule = Rule()
    agent = Agent(rule)
    env = gym.make(rule.env_name)
    reward_adjustment = True if rule.env_name == 'MountainCarContinuous-v0' else False
    score_list = []
    average_score_list = []
    for e in range(rule.n_episode):
        done = False
        score = 0
        state = env.reset()
        while not done:
            if rule.render:
                env.render()
            action = agent.get_action(state)
            state_, reward, done, _ = env.step(action)
            if reward_adjustment:
                reward = 1 / (0.5 - state_[0]) * abs(state_[1])
                reward = np.clip(reward, -1., 1.)
                if done and state_[0] > 0.45:
                    reward += 10
            score += reward
            agent.replaybuffer.store(state, action, reward, state_, done)
            agent.learn()
            state = state_
        if (e+1) % rule.save_cycle == 0:
            agent.save()
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        average_score_list.append(average_score)
        print(f'[{e+1}/{rule.n_episode}] [Score: {score:.0f}] [Average Score: {average_score:.1f}]')

    plt.plot(np.arange(rule.n_episode), average_score_list)
    plt.xlabel('n_episode')
    plt.ylabel('Moving Average Score')
    plt.title(rule.env_name)
    plt.show()
    
    