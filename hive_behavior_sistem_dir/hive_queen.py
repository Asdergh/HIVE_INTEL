import numpy as np
import torch as th
import random as rd


class HIVE_GUEEN():
    
    def __init__(self, random_walk_mode, trajectory_planing_mode, trajectory_path_cores, steps_count=10000, step_size=6) -> None:
        
        self.start_position = np.random.randint(-100, 100, size=3, dtype=np.float32)
        self.curent_position = np.zeros(3, dtype=np.float32)

        self.random_walk_mode = random_walk_mode
        self.trajectory_planing_mode = trajectory_planing_mode

        self.steps_count = steps_count
        self.step_size = step_size

        self.trajectory_path_cores = trajectory_path_cores
        self.step = 0
    
    def change_position(self):

        if self.random_walk_mode:
            
            self.curent_position[0] = self.curent_position[0] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
            self.curent_position[1] = self.curent_position[1] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
            self.curent_position[2] = self.curent_position[2] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
        
        elif self.trajecotry_planing_mode:

            self.curent_position = self.trajectory_path_cores[self.step]

        elif (self.random_walk_mode) and (self.trajectory_planing_mode):
            raise Exception("choice error: [both methods were selected]")
    
    def reset_position(self):
        
        self.curent_position = self.start_position
    



            


