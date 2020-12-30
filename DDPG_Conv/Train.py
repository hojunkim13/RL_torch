from Agent import Agent
import gym
from Rule import Rule
import matplotlib.pyplot as plt
import numpy as np
from Utils import Tools

def frame_skip_step(action):
    global bonus
    reward = 0
    for _ in range(rule.frame_skip):
        if rule.render:
                env.render()
        tmp_state, tmp_reward, done, _ = env.step(action)
        if tmp_reward > 1:
            bonus += 0.01
            tmp_reward += bonus
        reward += tmp_reward / rule.frame_skip
        if done:
            break
    return tmp_state, reward, done, None



if __name__ == "__main__":
    #load parameters & agent & tools ...
    rule = Rule()
    tool = Tools(rule)
    agent = Agent(rule, tool)
    bonus = 0


    #make env
    env = gym.make(rule.env_name)
    score_list = []
    average_score_list = []

    for e in range(rule.n_episode):
        #initialize
        done = False
        score = 0
        #reset env & make first state
        tmp_state = env.reset()
        tmp_state = tool.preprocessing_image(tmp_state)
        tool.add_to_tmp(tmp_state)
        for _ in range(rule.frame_stack-1):
            tmp_state, _, _, _ = frame_skip_step(env.action_space.sample())
            tmp_state = tool.preprocessing_image(tmp_state)
            tool.add_to_tmp(tmp_state)
        state = tool.get_state()
        while not done:
            #get action
            action = agent.get_action(state)
            
            #frame skip step & get new state
            tmp_state, reward, done, _ = frame_skip_step(action)
            
            tmp_state = tool.preprocessing_image(tmp_state)
            tool.add_to_tmp(tmp_state)
            state_ = tool.get_state()
            
            #calc score & store transition
            score += reward
            agent.replaybuffer.store(state, action, reward, state_, done)

            #agent learn & update current state 
            agent.learn()
            state = state_
        env.close()
        if (e+1) % rule.save_cycle == 0:
            agent.save()
        score_list.append(score)
        average_score = np.mean(score_list[-50:])
        average_score_list.append(average_score)
        print(
            f'[{e+1}/{rule.n_episode}] [Score: {score:.0f}] [Average Score: {average_score:.1f}]')

    plt.plot(np.arange(rule.n_episode), average_score_list)
    plt.xlabel('n_episode')
    plt.ylabel('Moving Average Score')
    plt.title(rule.env_name)
    plt.show()
