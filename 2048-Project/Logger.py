import logging

logger = logging.getLogger("MCTS")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler("MCTS.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



class MCTSLogger(logging.Logger):
    def __init__(self, name):
        super(MCTSLogger, self).__init__(name)
        self.setLevel(logging.INFO)
        
    
    def reset_handler(self, episode = None):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt="%y/%m/%d")
        file_handler = logging.FileHandler(f"MCTS_{episode}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        if len(self.handelrs) == 0:
            self.addHandler(file_handler)
        else:
            self.handlers[0] = file_handler
