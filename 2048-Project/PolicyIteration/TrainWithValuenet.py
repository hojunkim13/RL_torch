import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.AgentValue import Agent
from Logger import logger
from Environment.DosEnv import _2048
from Environment.Utils import calc_value

lr = 1e-3
batch_size = 256
n_sim = 50

n_episode = 10000

agent = Agent((16,4,4), 4, lr, batch_size, n_sim)
env = _2048()
#agent.load("2048")


def main():
    score_list = []
    for e in range(n_episode):        
        done = False
        grid = env.reset()
        score = 0
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
            if not agent.mcts.root_node.legal_moves[action]:
                print("warning")
            grid, reward, done, info = env.step(action, False)    
            score += reward
            agent.storeTransition(grid)
            agent.mcts.setRoot(grid, action)
        value = calc_value(grid)
        loss = agent.learn(value)

        #done
        if (e+1) % 10 == 0:
            agent.save("2048")
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}, Loss : {loss:.3f}")

if __name__ == "__main__":
    main()