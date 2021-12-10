import gym
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

# ENV_NAME = "CartPole-v1"
ENV_NAME = "LunarLander-v2"
env = gym.make(ENV_NAME)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

EPOCH = 1000

agent = Agent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=5e-4,
    gamma=0.99,
    mem_max=30000,
    eps_decay=0.999,
    batch_size=256,
)

if __name__ == "__main__":
    scores = []
    best_mean_score = -1e3
    for e in range(EPOCH):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.getAction(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            reward = -10 if done else reward
            agent.memory.stackMemory(state, action, reward, state_, done)
            agent.learn()
            state = state_
        scores.append(score)
        mean_score = np.mean(scores[-100:])
        if mean_score > best_mean_score:
            agent.save(ENV_NAME)
            best_mean_score = mean_score
        print(
            f"Episode : {e+1}, Score : {score:.0f}, Average: {mean_score:.1f} Epsilon : {agent.eps}"
        )

    plt.plot(range(EPOCH), scores)
    plt.title(f"{ENV_NAME}, DQN")

