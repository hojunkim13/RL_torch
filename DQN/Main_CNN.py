import gym
from Agent import Agent
import numpy as np
from Utils.stateChanger import stateChanger

ENV_NAME = "PongDeterministic-v4"
env = gym.make(ENV_NAME)

changer = stateChanger()
agent = Agent(
    state_dim=(1, 96, 96),
    action_dim=2,
    lr=3e-4,
    gamma=0.95,
    mem_max=50000,
    eps_decay=0.99,
    batch_size=64,
)

EPOCH = 100000

if __name__ == "__main__":
    scores = []
    for e in range(EPOCH):
        tmp_state = env.reset()
        changer.append(tmp_state)
        score = 0
        done = False
        state = changer.get()
        best_mean_score = -1e3
        while not done:
            action = agent.getAction(state)
            reward = 0
            for _ in range(4):
                tmp_state, tmp_reward, done, _ = env.step(action + 2)
                reward += tmp_reward
                if done:
                    break
            reward /= 4

            changer.append(tmp_state)
            state_ = changer.get()
            score += reward

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
