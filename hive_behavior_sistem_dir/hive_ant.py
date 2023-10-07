import numpy as np
import torch as th
import random as rd


#----------------------------------#
# HIVE ANT 
#----------------------------------#
class HIVE_ANT():
    
    # variable ant_id: id номер муровья
    # type ant_id: strin
    # variable start_position: вектор стартовой позиции
    # type start_position: numpy ndarray
    # variable curent_position: вектор текущей позиции
    # type curent_position: numpy ndarray

    def __init__(self) -> None:
        
        self.ant_id = f"#$DR{rd.choice(['a', 'b', 'c', 'd', 'f'])}{rd.choice(['a', 'b', 'c', 'd', 'f'])}{rd.choice(['a', 'b', 'd', 'f'])}N#$"

        self.start_position = np.random.randint(-100, 100, size=3)
        self.curent_position = self.start_position
    
    # get hive member position
    def get_position(self):
        return self.curent_position
    
    # change the hive member position
    def change_position(self, new_position_vector):
        self.curent_position = new_position_vector
    
    # reset the hive member position
    def reset_position(self):
        self.curent_position = self.start_position

