import numpy as np
import torch as th
import matplotlib.pyplot as plt
import random as rd
import json as js
import statistics as stat
import pandas as pd

from matplotlib.animation import FuncAnimation
plt.style.use("dark_background")


    
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
    
    def __init__(self, random_walk_mode, trajectory_planing_mode=None, trajectory_path_cores=None, steps_count=10000, step_size=6) -> None:
        
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
        
        elif self.trajectory_planing_mode:

            self.curent_position = self.trajectory_path_cores[self.step]

        elif (self.random_walk_mode) and (self.trajectory_planing_mode):
            raise Exception("choice error: [both methods were selected]")
    
    # reset hive queen position
    def reset_position(self):
        
        self.curent_position = self.start_position
    
    # get hive queen position
    def get_position(self):

        return self.curent_position
    
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


#------------------------------------#
#INVIRONMENT
#------------------------------------#
class ENVIRONMENT():

    # в __init__ мы также задаем начальное состояние нашей среды
    def __init__(self, swarm_size, action_step_size=12.67) -> None:
        
        # variable swarm_size: колличество муравьев в улье
        # type swarm_size: int
        # variable hive_members_connections: словарь с текущими связями муравьев относительно друг друга
        # type hive_members_connections: python dict_object
        # variable hive_members: словарь с информацией о муровьях
        # type hive_members: python dict_object
        # variable hive_queen: королева улья
        # type hive_queen: HIVE_QUEEN_OBJECT
        # variable reward: вознограждение агента для поощрения и наказания для успешных и не успешных действий соответственно
        # type reward: int
        # variable total_reward: суммарное вознограждение
        # type total_reward: int
        # variable rewards_list: лист с данными вознограждений
        # type rewards_list: python list_object
        # variable states_list: лист с данными состояний в нашем случае (среднее расстояние от королевы до муравья)
        # type states_list: python list_object
        # variable need_range: радиус орбиты (радиус сферы на которой около которой должны летать муровьи)
        # type need_range: float

        self.swarm_size = swarm_size
        self.hive_members_connections = {}
        self.hive_members = {}
        self.hive_queen = HIVE_QUEEN(trajectory_path_cores=True, random_walk_mode=False)

        self.reward = 1.0
        self.total_reward = 0.0
        self.rewards_list = []
        self.states_list = []

        self.need_range = 45.67
        self.ant_neighbours_need_range = 12.4
        self.step_number = 0

        self.action_step_size = action_step_size


        for ant in range(self.swarm_size):
            hive_ant = HIVE_ANT()
            self.hive_members[f"{hive_ant.ant_id}"] = hive_ant

        for ant in self.hive_members.keys():
            for sub_ant in self.hive_members.keys():

                if ant != sub_ant:

                    self.hive_members_connections[ant] = {
                        f"neightbour ant id: {sub_ant}": 
                        {
                            "neighbour ant": self.hive_members[ant],
                            "distance": self.hive_members[ant].get_position() - self.hive_members[sub_ant].get_position()
                        }
                    }

                else:
                    pass
    
        self.hive_distances_database = pd.DataFrame(columns=[self.hive_members[ant].ant_id for ant in self.hive_members.keys()])
        self.past_mean_distance = 0.0
    
    def get_curent_hive_info(self):

        self.ant_connection_database = pd.from_dict(self.hive_members_connections)
        self.ant_


    # вернуть среду в значальное положение
    def reset(self):

        for ant in self.hive_members.keys():
            self.hive_members[ant].reset_position()
    
    # расчет состояния среды представленного в дистанции между королевой улья и муровьем
    # расчет возногрождения r дикретного:
    # r = 1.0 если {E[t] < E[t - 1] and E[t] > R} или {E[t] > E[t - 1] and E[t] < R}
    # r = -1.0 если {E[t] < E[t - 1] and E[t] < R} или {E[t] > E[t - 1] and E[t] > R}
    # где E[t] -- математическое ожидание на тукущей итерации; E[t - 1] -- математическое ожидение на предыдущей итерацииж; R -- допустимый радиус орбиты
    def step(self):
        
        self.mean_distance = stat.mean(
            [

                np.sqrt(self.hive_members[ant].get_position()[0] ** 2 + self.hive_members[ant].get_position()[1] ** 2 + self.hive_members[ant].get_position()[0] ** 2) 
                - np.sqrt(self.hive_queen.get_position()[0] ** 2 + self.hive_queen.get_position()[1] ** 2 + self.hive_queen.get_position()[2] ** 2) 
                for ant in self.hive_members.keys()
            
            ])
        
        if (self.mean_distance > self.need_range) and (self.mean_distance < self.past_mean_distance):
            self.reward = 1.0

        elif (self.mean_distance > self.need_range) and (self.mean_distance > self.past_mean_distance):
            self.rewards = -1.0
        
        elif (self.mean_distance < self.need_range) and (self.mean_distance > self.past_mean_distance):
            self.reward = 1.0
        
        elif (self.mean_distance < self.need_range) and (self.mean_distance < self.past_mean_distance):
            self.reward = -1.0

        if (self.reward == 1.0) and (self.mean_distance > self.need_range):

            for ant in self.hive_members.keys():

                new_cores = self.hive_members[ant].get_position() - np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)

        
        elif (self.reward == 1.0) and (self.mean_distance < self.need_range):

            for ant in self.hive_members.keys():
                
                new_cores_rel_queen = self.hive_members[ant].get_position() + np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)
                
                for ant_neighbour in self.hive_members_connections[ant]:

                    self.neighbour_distance_len = np.sqrt(ant_neighbour["distance"][0] ** 2 + ant_neighbour["distance"][1] ** 2 + ant_neighbour["distance"][2] ** 2)
                    if self.neighbour_distance_len < self.neighbours_need_range:
                        self.neighbour_new_cores = ant_neighbour["distance"] + 3.3412
                        self.ant_neighbour["neighbour ant"].change_position(self.neighbour_new_cores)




        elif (self.reward == -1.0) and (self.mean_distance > self.need_range):
            
            for ant in self.hive_members.keys():
                
                new_cores_rel_queen = self.hive_members[ant].get_position() + np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)

                for ant_neighbour in self.hive_members_connections[ant]:

                    self.neighbour_distance_len = np.sqrt(ant_neighbour["distance"][0] ** 2 + ant_neighbour["distance"][1] ** 2 + ant_neighbour["distance"][2] ** 2)
                    if self.neighbour_distance_len < self.neighbours_need_range:
                        self.neighbour_new_cores = ant_neighbour["distance"] + 3.3412
                        self.ant_neighbour["neighbour ant"].change_position(self.neighbour_new_cores)

        elif (self.reward == -1.0) and (self.mean_distance < self.need_range):

            for ant in self.hive_members.keys():
                
                new_cores_rel_queen = self.hive_members[ant].get_position() + np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)

                for ant_neighbour in self.hive_members_connections[ant]:

                    self.neighbour_distance_len = np.sqrt(ant_neighbour["distance"][0] ** 2 + ant_neighbour["distance"][1] ** 2 + ant_neighbour["distance"][2] ** 2)
                    if self.neighbour_distance_len < self.neighbours_need_range:
                        self.neighbour_new_cores = ant_neighbour["distance"] + 3.3412
                        self.ant_neighbour["neighbour ant"].change_position(self.neighbour_new_cores)
        
        self.states_list.append(self.mean_distance)
        self.rewards_list.append(self.reward)
        self.total_reward += self.reward
        self.hive_distances_database.loc[f"epizode_number:{self.step_number}"] = [

                np.sqrt(self.hive_members[ant].get_position()[0] ** 2 + self.hive_members[ant].get_position()[1] ** 2 + self.hive_members[ant].get_position()[0] ** 2) 
                - np.sqrt(self.hive_queen.get_position()[0] ** 2 + self.hive_queen.get_position()[1] ** 2 + self.hive_queen.get_position()[2] ** 2) 
                for ant in self.hive_members.keys()
            
            ]
        self.step_number += 1
        self.past_mean_distance = self.mean_distance

        print(self.past_mean_distance)
        print(self.hive_distances_database)
        return self.mean_distance, self.reward
        

# TODO name of all variables was wrong so we need to accumulate there positions to ppload new targets so actualy we need to
if __name__ == "__main__":

    env = ENVIRONMENT(swarm_size=100, action_step_size=12.12)
    figure = plt.figure()
    view_3d = figure.add_subplot(projection="3d")

    def animation(curent_step):

        view_3d.clear()
        env.hive_queen.change_position()
        state, reward = env.step()
        print(f"state: [{state}], reward: [{reward}], total_reward: [{env.total_reward}]")

        ants_position_x_vector = np.array([env.hive_members[ant_id].get_position()[0] for ant_id in env.hive_members.keys()])
        ants_position_y_vector = np.array([env.hive_members[ant_id].get_position()[1] for ant_id in env.hive_members.keys()])
        ants_position_z_vector = np.array([env.hive_members[ant_id].get_position()[2] for ant_id in env.hive_members.keys()])
        hive_queen_position = env.hive_queen.get_position()

        view_3d.scatter(ants_position_x_vector, ants_position_y_vector, ants_position_z_vector, c=ants_position_z_vector, cmap="twilight", s=43.56, alpha=0.34)
        view_3d.scatter(hive_queen_position[0], hive_queen_position[1], hive_queen_position[2], color="black", s=100)
        view_3d.quiver(hive_queen_position[0], hive_queen_position[1], hive_queen_position[2], 32.12, 0, 0, color="blue")
        view_3d.quiver(hive_queen_position[0], hive_queen_position[2], hive_queen_position[2], 0, 32.12, 0, color="red")
        view_3d.quiver(hive_queen_position[0], hive_queen_position[2], hive_queen_position[2], 0, 0, 32.12, color="green")

    demo = FuncAnimation(figure, animation, interval=1)
    plt.show()
    

