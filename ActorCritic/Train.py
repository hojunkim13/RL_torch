import gym
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 1e-4
gamma = 0.99
EPOCHS = 500

render = False
load = False
SAVE_FREQ = 100
PATH = "weights/"

if __name__ == "__main__":
    agent = Agent(ENV_NAME, state_dim, action_dim, lr, gamma)
    if load:
        agent.load(PATH)
    score_list = []

    for e in range(EPOCHS):
        done = False
        score = 0
        state = env.reset()
        while not done:
            if render:
                env.render()
            action = agent.getAction(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            if ENV_NAME == "MountainCar-v0":
                if done and state_[0] > 0.45:
                    reward += 1
            elif ENV_NAME == "CartPole-v1":
                if done:
                    reward = -100

            agent.train(state, action, reward, state_, done)
            state = state_
        # Episode is done.
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        print(
            f"[{e+1}/{EPOCHS}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]"
        )
        if (e + 1) % SAVE_FREQ == 0:
            agent.save(PATH, e + 1)

    env.close()
    plt.plot(range(EPOCHS), score_list)
