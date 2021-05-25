import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PolicyIteration.Agent import Agent
from Environment.DosEnv import _2048
from Environment.Utils import preprocessing
from collections import deque

lr = 1e-4
batch_size = 256
n_sim = 100
maxlen = 50000
n_episode = 10000
state_dim = (16,4,4)
action_dim = 4
agent = Agent(state_dim, action_dim, lr, batch_size, n_sim, maxlen)
env = _2048()
#agent.load("2048")


def main():
    score_list = []    
    for e in range(n_episode):        
        done = False
        grid = env.reset()
        score = 0
        loss = 0        
        while not done:
            #env.render()
            action = agent.getAction(grid)
            if not agent.mcts.root_node.legal_moves[action]:
                raise SystemError("Agent did illegal action")
            new_grid, reward, done, info = env.step(action)
            agent.storeTranstion(preprocessing(grid), reward)
            grid = new_grid
            score += reward
        agent.pushMemory()
        loss = agent.learn()
        if (e+1) % 10 == 0:
            agent.save("2048")
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}, Loss : {loss:.3f}")

if __name__ == "__main__":
    main()