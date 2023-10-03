import numpy as np
import torch as th
import matplotlib.pyplot as plt
import random as rd
import json as js
import statistics as stat


#-----------------------------------------#
# HIVE QUEEN CLASS
#-----------------------------------------#
class HIVE_QUEEN():
    
    def __init__(self, random_walk_mode, trajectory_planing_mode=None, trajectory_path_cores=None, steps_count=10000, step_size=6) -> None:
        
        self.start_position = np.random.randint(-100, 100, size=3)
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
    
    def get_position(self):

        return self.curent_position

#----------------------------------#
# HIVE ANT 
#----------------------------------#
class HIVE_ANT():
    
    def __init__(self) -> None:
        
        self.ant_id = f"#$DR{rd.choice(['a', 'b', 'c', 'd', 'f'])}{rd.choice(['a', 'b', 'c', 'd', 'f'])}{rd.choice(['a', 'b', 'd', 'f'])}N#$"

        self.start_position = np.random.randint(-100, 100, size=3)
        self.curent_position = self.start_position
        
    def get_position(self):
        return self.curent_position

    def change_position(self, new_position_vector):
        self.curent_position = new_position_vector
    
    def reset_position(self):
        self.curent_position = self.start_position

#------------------------------------#
#INVIRONMENT
#------------------------------------#
class INVIRONMENT():

    def __init__(self, swarm_size) -> None:
        
        self.swarm_size = swarm_size
        self.hive_members_connections = {}
        self.hive_members = {}
        self.hive_queen = HIVE_QUEEN(random_walk_mode=True)

        self.state = []
        self.reward = 0.0
        self.rewars_list = []


        for ant in range(self.swarm_size):
            hive_ant = HIVE_ANT()
            self.hive_members[f"{hive_ant.ant_id}"] = hive_ant

        for ant in self.hive_members.keys():
            for sub_ant in self.hive_members.keys():

                if ant != sub_ant:

                    self.hive_members_connections[ant] = {
                        f"neiborhood ant |====|{sub_ant}": 
                        {
                            "nb ant": self.hive_members[sub_ant],
                            "distance": self.hive_members[ant].get_position() - self.hive_members[sub_ant].get_position()
                        }
                    }

                else:
                    pass

        self.mean_distance = stat.mean([np.sqrt(self.hive_members[ant].get_position()[0] ** 2 + self.hive_members[ant].get_position()[1] ** 2 + self.hive_members[ant].get_position()[0] ** 2) 
                                        - np.sqrt(self.hive_queen.get_position()[0] ** 2 + self.hive_queen.get_position()[1] ** 2 + self.hive_queen.get_position()[2] ** 2) for ant in self.hive_members.keys()])
        print(self.mean_distance)

    def reset(self):

        for ant in self.hive_members.keys():
            self.hive_members[ant].reset_position()
    
    def step(self):
        pass




        

        

                
                

        
    
if __name__ == "__main__":
    env = INVIRONMENT(swarm_size=12)


