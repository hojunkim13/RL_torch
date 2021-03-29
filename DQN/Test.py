import os, sys
import gym
from Agent import Agent
import numpy as np
import torch

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env._max_episode_steps = 1e+7
    agent = Agent(n_state = 4, n_action= 2)
    agent.net.eval()
    agent.net_.eval()

    if not os.path.isdir("model"):
        print("Can't find model weights")
        sys.exit()
    saved = False
    for file_name in os.listdir("model"):
        if env_name in file_name:
            saved = True
            path = os.path.abspath("model/" + file_name)
            weights = torch.load(path)
            agent.net.load_state_dict(weights)
            agent.sync()
    if not saved:
        print("Can't find model weights")
        sys.exit()
        
    state = env.reset()
    score = 0
    done = False
    while not done:
        env.render()
        action = agent.getAction(state, True)
        if np.random.rand() <= 0.1:
            action = int(not action)
        state, reward, done, _ = env.step(action)
        score += reward                                                        
    print(f"Score : {score}")
    
    

            
