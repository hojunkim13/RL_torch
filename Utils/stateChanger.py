import numpy as np
from collections import deque
import cv2

class stateChanger:
    def __init__(self):
        self.states = deque(maxlen=2)
    
    def append(self, state):
        state = self.preprocessing(state)
        if len(self.states) == 0:
            self.states.append(state)
        self.states.append(state)
            
    def get(self):
        state_difference = self.states[0] - self.states[1]
        state_difference = np.transpose(state_difference, (2,0,1,))
        return state_difference
    
    @staticmethod
    def preprocessing(state):        
        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        if state_gray.shape[:-1] != (96, 96):
            state_gray = state_gray[35:190]
            state = cv2.resize(state_gray, dsize=(96, 96), interpolation=cv2.INTER_AREA)        
        if state.max() >= 255:
            state = state / 255.0        
        state = state.reshape(96,96,1)
        return state