import gym

env = gym.make('MountainCarContinuous-v0')

for _ in range(10):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        print(action)
        obs_, reward, done, _ = env.step(action)
        score += reward
    
    print(score)