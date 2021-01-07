from Agent import Agent
import gym
import numpy as np
from Environment import Environment




env_name = 'CartPole-v1'


n_episode = 1000


env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
load = False
render = False
save_freq = 100
lr = 1e-4
gamma = 0.99
lmbda = 0.95
epsilon = 0.1

buffer_size = 1000
batch_size = 256
k_epochs = 10

path = 'model/' + env_name

agent = Agent(state_dim, action_dim, lr, gamma, lmbda, epsilon, buffer_size,batch_size, k_epochs)


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
                action, log_prob, h_out = agent.get_action(state)
                state_, reward, done, _ = env.step(action)
                score += reward
                agent.store(state, action, log_prob, reward, state_, done, h_in, h_out)
                state = state_
                agent.learn()            
            
        #done
        env.close()
        if (e+1) % save_freq == 0:
            agent.save(path)
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

