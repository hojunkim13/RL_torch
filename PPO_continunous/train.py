import gym
import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np



env_name = 'LunarLander-v2'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim= env.action_space.n
n_episode = 2000
lr = 5e-4
gamma = 0.98
lmbda = 0.95
epsilon = 0.1
timestep = 20
k_epochs = 3

agent = Agent(state_dim, action_dim, lr,epsilon, gamma, lmbda, timestep, k_epochs)


if __name__ == "__main__":
    score_list = []
    avg_score_list = []
    for e in range(n_episode):
        score = 0
        done = False
        state = env.reset()
        while not done:
            for _ in range(timestep):
                #env.render()
                action, log_prob = agent.get_action(state)
                state_, reward, done, _ = env.step(action)
                score += reward
                agent.store((state,action,log_prob,reward,state_,done))
                state = state_
                if done:
                    break
            agent.learn()
            
        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}], [Average Score: {avg_score:.1f}]')
    env.close()
    
    plt.plot(avg_score_list)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Score')
    plt.title(env_name)
    plt.show()
