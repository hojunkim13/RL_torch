import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.AgentValue import Agent
from Logger import logger
from Environment.DosEnv import _2048
from Environment.Utils import calc_value, preprocessing
from collections import deque

lr = 5e-5
batch_size = 256
n_sim = 100
maxlen = 10000
n_episode = 10000
state_dim = (16,4,4)
action_dim = 4
agent = Agent(state_dim, action_dim, lr, batch_size, n_sim, maxlen)
env = _2048()
agent.load("2048")


def main():
    score_list = []    
    for e in range(n_episode):        
        done = False
        grid = env.reset()
        tmp_memory = deque()
        score = 0
        agent.step_count = 0        
        log = (e) % 10 == 10
        if log:
            logger.info("########################")
            logger.info(f"##### EPISODE {e+1}#####")
            logger.info("########################")
            logger.info("########################")
        while not done:
            #env.render()
            tmp_memory.append(preprocessing(grid))
            agent.mcts.reset(grid)
            action = agent.getAction(log)
            if not agent.mcts.root_node.legal_moves[action]:
                print("warning")
            grid, reward, done, info = env.step(action)    
            score += reward            
        outcome = calc_value(grid)
        agent.pushMemory(tmp_memory, outcome)
        loss = agent.learn()

        #done
        if (e+1) % 10 == 0:
            agent.save("2048")
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}, Loss : {loss:.3f}")

if __name__ == "__main__":
    main()