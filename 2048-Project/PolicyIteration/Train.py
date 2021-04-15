import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.Agent import Agent
import numpy as np
from Environment.DosEnv import _2048

env_name = "2048"
env = _2048()
state_dim = (16,4,4)
action_dim = 4

n_episode = 10000
load = True
save_freq = 10
lr = 1e-2
batch_size = 256
agent = Agent(state_dim, action_dim, lr, batch_size)


if load:
    agent.load(env_name)

def main():
    score_list = []
    for e in range(n_episode):
        done = False
        grid = env.reset()
        agent.step_count = 0
        while not done:
            action = agent.getAction(grid)
            agent.storeTransition(grid, action)
            grid, _, done, info = env.step(action, False)    
            agent.step_count += 1
        outcome = np.log2(info) / 10        
        agent.learn(outcome = outcome)

        #done
        if (e+1) % save_freq == 0:
            agent.save(env_name)
        score_list.append(info)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Max Tile : {info}, Average: {average_score:.1f}")


if __name__ == "__main__":
    main()