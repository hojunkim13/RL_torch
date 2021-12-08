import gym
from Agent import Agent
import numpy as np
from Train import ENV_NAME, PATH, state_dim, action_dim

EPOCHS = 10

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME, state_dim, action_dim, 1, 1, 1)
    agent.load(PATH, 3000)
    score_list = []

    for e in range(EPOCHS):
        done = False
        score = 0
        state = env.reset()
        while not done:
            env.render()
            action = agent.getAction(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            state = state_
        # Episode is done.
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        print(
            f"[{e+1}/{EPOCHS}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]"
        )
    env.close()
