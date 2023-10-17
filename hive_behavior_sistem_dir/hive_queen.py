import numpy as np
import torch as th
import random as rd


class HIVE_QUEEN():
    
    # variable start_position: вектор стартового положения королевы
    # type start_position: numpy ndarray
    # variable curent_position: вектор текущего положения королевы
    # type curent_position: numpy ndarray
    # variable random_walk_mode: мод случайного блуждания
    # type random_walk_mode: bool
    # variable trajectory_planing_mode: мод движение по траектории
    # type trajectory_planing_mode: bool
    # variable steps_count: количество шагов движение королевы
    # type steps_count: int
    # variable step_size: длинна рандомного смещения
    # type step_size: flaot
    # variable trajectory_path_cores: матриа с координатами траектории
    # type trajectory_path_cores: numpy ndarray
    # variable step: шаг итерации
    # type step: int
    
    def __init__(self, random_walk_mode=False, trajectory_planing_mode=False, trajectory_path_cores=None, steps_count=10000, step_size=6) -> None:
        
        self.start_position = np.random.randint(-100, 100, size=3)
        self.curent_position = np.zeros(3, dtype=np.float32)

        self.random_walk_mode = random_walk_mode
        self.trajectory_planing_mode = trajectory_planing_mode

        self.steps_count = steps_count
        self.step_size = step_size

        self.trajectory_path_cores = trajectory_path_cores
        self.step = 0
    
    # change hive queen position
    def change_position(self):

        if self.random_walk_mode:
            
            self.curent_position[0] = self.curent_position[0] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
            self.curent_position[1] = self.curent_position[1] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
            self.curent_position[2] = self.curent_position[2] + rd.choice([-1, 1]) * rd.randint(-self.step_size, self.step_size)
        
        elif self.trajecotry_planing_mode:

            self.curent_position[0] = np.cos(self.step * np.pi) * np.sin(self.step * np.pi)
            self.curent_position[1] = np.sin(self.step * np.pi) + np.cos(self.step * np.pi)
            self.curent_position[2] = self.step

        elif (self.random_walk_mode) and (self.trajectory_planing_mode):

            raise Exception("choice error: [both methods were selected]")
    
    # reset hive queen position
    def reset_position(self):
        
        self.curent_position = self.start_position
    
    # get hive queen position
    def get_position(self):

        return self.curent_position

    



            


