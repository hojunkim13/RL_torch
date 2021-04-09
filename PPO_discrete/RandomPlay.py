import numpy as np
from AtariWrapper import Environment
import time

env_name = "Breakout-v0"
n_episode = 100
render = True

env = Environment(env_name)

if __name__ == "__main__":
    score_list = []
    mas_list = []
    for e in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        while not done:
            if render:
                env.render()
                time.sleep(0.02)
            state_, reward, done, _ = env.step(env.env.action_space.sample(), render)
            score += reward
            state = state_
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        mas_list.append(average_score)
        print(f'[{e+1}/{n_episode}] [Score: {score:.1f}] [Average Score: {average_score:.1f}]')

