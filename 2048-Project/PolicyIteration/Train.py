import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.Agent import Agent
import numpy as np
from Environment.DosEnv import _2048
from Environment.Utils import calc_value
from Logger import logger

env_name = "2048"
env = _2048()
state_dim = (16,4,4)
action_dim = 4

n_episode = 10000
load = False
save_freq = 10
lr = 1e-3
batch_size = 128
n_sim = 100


agent = Agent(state_dim, action_dim, lr, batch_size, n_sim)

if load:
    agent.load(env_name)

def main():
    score_list = []
    for e in range(n_episode):
        done = False
        grid = env.reset()
        agent.step_count = 0
        log = (e) % 10 == 0
        if log:
            logger.info(f"EPISODE {e+1} ##########")
            logger.info("########################")
            logger.info("########################")
            logger.info("########################")
        while not done:
            action, mcts_probs = agent.getAction(grid, log)
            grid, _, done, info = env.step(action, False)    
            agent.storeTransition(grid, mcts_probs) 
        agent.learn()

        #done
        if (e+1) % save_freq == 0:
            agent.save(env_name)
        score_list.append(info)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Max Tile : {info}, Average: {average_score:.1f}")


if __name__ == "__main__":
    main()