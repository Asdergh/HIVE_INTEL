import numpy as np
import torch as th
import random as rd


class HIVE_ANT():
    
    def __init__(self) -> None:
        

        self.ant_id = f"#${rd.chioce(['A', 'B', 'D', 'F'])}\
        {rd.choice(['A', 'B', 'D', 'F'])}{rd.choice(['a', 'b', 'c', 'd', 'f'])}\
            {rd.choice(['a', 'b', 'c', 'd', 'f'])}{rd.choice(['a', 'b', 'd', 'f'])}\
                {rd.choice(['A', 'B', 'C', 'D', 'F'])}#$"
        
        self.start_position = np.random.randint(-100, 100, size=3, dtype=np.float32)
        self.curent_position = np.zeros(3)
        
    def get_position(self):
        return self.curent_position

    def change_position(self, new_position_vector):
        self.curent_position = new_position_vector
    
    def reset_position(self):
        self.curent_position = self.start_position

