import gym
from Agent import Agent

ENV_NAME = "LunarLander-v2"  # can be CartPole-v1
env = gym.make(ENV_NAME)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Agent(state_dim, action_dim, 1, 1, 1, 1, 1)
agent.load(ENV_NAME)
EPOCH = 10

if __name__ == "__main__":

    for e in range(EPOCH):
        state = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            action = agent.getAction(state, True)
            state, reward, done, _ = env.step(action)
            score += reward
        print(f"Episode : {e+1}, Score : {score}")
