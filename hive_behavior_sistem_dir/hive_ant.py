import numpy as np
import torch as th
import random as rd


class HIVE_ANT():
    
    def __init__(self) -> None:
        
        self.id_low_register = ["abcdf"]
        self.id_hight_register = ["ABCDF"]
        self.id_number = ["12345"]
        self.ant_id = f"#${rd.chioce(self.id_hight_register)}{rd.choice(self.id_hight_register)}{rd.choice(self.id_low_register)}\
                            {rd.choice(self.id_low_register)}{rd.choice(self.id_low_register)}{rd.choice(self.id_hight_register)}#$"

        self.start_position = np.random.randint(-100, 100, size=3, dtype=np.float32)
        self.curent_position = np.zeros(3)
        
    def get_cores(self):
        return self.curent_position

    def change_position(self, new_position_vector):
        self.curent_position = new_position_vector
    
    def reset_position(self):
        self.curent_position = self.start_position

