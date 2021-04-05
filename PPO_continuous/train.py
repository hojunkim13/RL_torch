import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np
#from CarRacingWrapper import Environment
from PongWrapper import Environment


#env_name = 'CarRacing-v0'
env_name = 'Pong-v0'

env = Environment()
state_dim = (1,96,96)
action_dim = 3

save_cycle = 100
load = False
render = True
n_episode = 10000
lr = 1e-3
gamma = 0.99
lmbda = 0.95
epsilon = 0.2
buffer_size = 1000
batch_size = 512
k_epochs = 10
path = './model/' + env_name
agent = Agent(state_dim, action_dim, lr,epsilon, gamma, lmbda, buffer_size, batch_size, k_epochs)


if __name__ == "__main__":
    if load:
        agent.load(path)
    score_list = []
    avg_score_list = []
    for e in range(n_episode):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action, log_prob = agent.get_action(state.cuda())
            state_, reward, done, _ = env.step(action, render)
            score += reward
            agent.store((state,action,log_prob,reward,state_,done))
            state = state_
            agent.learn()
        env.close()
        if (e+1) % save_cycle ==0:
            agent.save(path)            
        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}], [Average Score: {avg_score:.1f}]')
    #env.close()    
    plt.plot(avg_score_list)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Score')
    plt.title(env_name)
    plt.show()

