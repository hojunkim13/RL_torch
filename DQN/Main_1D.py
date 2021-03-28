import os, re
import gym
from Agent import Agent
import numpy as np
import torch

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = Agent(n_state = 4, n_action= 2)
    agent.load(env_name)
    n_episode = 300
    scores = []
    for e in range(n_episode):
        state = env.reset()
        score = 0
        done = False
        while not done:
            #env.render()
            action = agent.getAction(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done else -100
            agent.memory.stackMemory(state, action, reward, state_, done)
            agent.learn()
            state = state_
        agent.sync()
        scores.append(score)
        if len(set(scores[-10:])) == 1 and len(scores) != 1:
            print("BREAK!!")
            break
        movingAverageScore = np.mean(scores[-100:])
        print(f"Episode : {e}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")    
    if not os.path.isdir("model"):
        os.mkdir("model")
    model_list = os.listdir("model")
    saved = False
    score_now = int(movingAverageScore)
    for file_name in model_list:
        if env_name in file_name:
            saved = True
            try:
                score_old = re.match(r"\d+.pt", file_name).group()      
            except:
                score_old = 0
                  
            if score_now >= score_old:
                torch.save(agent.net_.state_dict(), f"./model/{env_name}_{score_now}.pt")
                os.remove("model/" + file_name)
    
    if not saved:
        torch.save(agent.net_.state_dict(), f"./model/{env_name}_{score_now}.pt")

            
