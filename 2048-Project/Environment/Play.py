from DosEnv import _2048
from Utils import *

env_name = "2048"
env = _2048()

if __name__ == "__main__":
    retry = True
    while retry:        
        done = False
        score = 0
        grid = env.reset()
        while not done:
            env.render()
            
            try:
                action = input("\n2,4,8,6 입력하세요 (하, 좌, 상, 우)\n")
                action = {"4":0, "8":1, "6":2, "2":3, "5":3}[action]
            except KeyError:
                action = input("\n2,4,8,6 입력하세요 (하, 좌, 상, 우)\n")
                action = {"4":0, "8":1, "6":2, "2":3, "5":3}[action]
            grid, reward, done, info = env.step(action)            
            score += reward        
        response = input(f"게임 종료! [{score}]점\n계속하시겠습니까? (y/n)\n")
        if response == "n":
            retry = False
        
        
