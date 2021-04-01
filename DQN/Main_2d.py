import gym
from Agent import Agent
import numpy as np
import torch
from Utils.stateChanger import stateChanger

if __name__ == "__main__":
    env_name = "Pong-v0"
    env = gym.make(env_name)    
    changer = stateChanger()
    agent = Agent(n_state = (1,96,96), n_action= 2, lr = 1e-4, gamma = 0.99, mem_max = 30000,
                epsilon_decay = 0.999, batch_size = 128)
    
    actionSpace = [0]
    n_episode = 200
    scores = []
    for e in range(n_episode):
        tmp_state = env.reset()
        changer.append(tmp_state)
        score = 0
        done = False
        state = changer.get()
        while not done:
            env.render()
            action = agent.getAction(state)
            reward = 0
            for _ in range(4):
                tmp_state, tmp_reward, done, _ = env.step(action + 2)
                reward += tmp_reward
                if done:
                    break
            reward /= 4

            changer.append(tmp_state)
            state_ = changer.get()
            score += reward

            agent.storeTransition(state, action, reward, state_, done)
            agent.learn()
            state = state_
        #env.close()
        agent.sync()
        try:
            if max(scores) <= score:
                torch.save(agent.net_.state_dict(), f"./DQN//model/{env_name}.pt")
        except:
            pass
        scores.append(score)
        movingAverageScore = np.mean(scores[-100:])
        print(f"Episode : {e+1}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")
        
            
