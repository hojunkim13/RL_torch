import gym
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 3e-4
gamma = 0.99
n_step = 300
EPOCHS = 3000


render = False
load = False
SAVE_FREQ = 100
PATH = "weights/"


def moving_average(score, n=3):
    ret = np.cumsum(score, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


if __name__ == "__main__":
    agent = Agent(ENV_NAME, state_dim, action_dim, lr, gamma, n_step)
    if load:
        agent.load(PATH)
    scores = []
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
            agent.storeTransition(state, action, reward, state_, done)
            agent.train()
            state = state_
        # Episode is done.
        scores.append(score)
        aver_score = np.mean(scores[-20:])
        print(
            f"[{e+1}/{EPOCHS}] [Score: {score:.1f}] [Average Score: {aver_score:.1f}]"
        )
        if (e + 1) % SAVE_FREQ == 0:
            agent.save(PATH, e + 1)

    env.close()
    # Plot
    fig, ax = plt.subplots()
    ax.plot(scores, label="score")
    ax.plot(moving_average(scores), label="moving average")
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("A2C on " + ENV_NAME)
    plt.show()

