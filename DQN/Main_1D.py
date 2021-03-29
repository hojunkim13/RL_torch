import gym
from Agent_noPER import Agent
import numpy as np

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = Agent(n_state = 4, n_action= 2, lr = 1e-3, gamma = 0.99, mem_max = 10000,
                epsilon_decay = 0.999, batch_size = 64)
    #agent.load(env_name)
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
        movingAverageScore = np.mean(scores[-100:])
        print(f"Episode : {e}, Score : {score:.0f}, Average: {movingAverageScore:.1f} Epsilon : {agent.epsilon}")    
    plt.plot(range(n_episode), scores)
    plt.title(f"{env_name}, DQN")

            
