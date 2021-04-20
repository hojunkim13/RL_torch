import numpy as np
import sys
sys.path.append('c:\\Users\\KHJ\\Desktop\\deeplearn\\Reinforcement\\torch\\2048-Project')
from PolicyIteration.AgentValue import Agent
from Logger import logger
from Environment.DosEnv import _2048
from Environment.Utils import calc_value

lr = 1e-3
batch_size = 256
n_sim = 100

n_episode = 1000

agent = Agent((16,4,4), 4, lr, batch_size, n_sim)
env = _2048()


def main():
    score_list = []
    for e in range(n_episode):
        
        done = False
        grid = env.reset()
        agent.step_count = 0
        agent.mcts.setRoot(grid)
        log = (e) % 10 == 0
        if log:
            logger.info(f"EPISODE {e+1} ##########")
            logger.info("########################")
            logger.info("########################")
            logger.info("########################")
        while not done:
            action = agent.getAction(log)
            grid, _, done, info = env.step(action, True)    
            agent.storeTransition(grid) 
            agent.mcts.setRoot(grid, action)
        value = calc_value(grid)
        agent.learn(value)

        #done
        if (e+1) % 10 == 0:
            agent.save("2048")
        score_list.append(info)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Max Tile : {info}, Average: {average_score:.1f}")

if __name__ == "__main__":
    main()