import numpy as np
import torch as th
import matplotlib.pyplot as plt
import random as rd
import json as js
import statistics as stat
import pandas as pd


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
        
        elif self.trajecotry_planing_mode:

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
        self.hive_queen = HIVE_QUEEN(random_walk_mode=True)

        self.reward = 1.0
        self.total_reward = 0.0
        self.rewards_list = []
        self.states_list = []

        self.need_range = 45.67
        self.step_number = 0

        self.action_step_size = action_step_size


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
        
        self.hive_distances_database = pd.DataFrame(columns=[self.hive_members[ant].ant_id for ant in self.hive_members.keys()])
        self.past_mean_distance = 0.0
    
    # reset the state
    def reset(self):

        for ant in self.hive_members.keys():
            self.hive_members[ant].reset_position()
    
    #change stats for learning
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
                new_cores = self.hive_members[ant].get_position() + np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)
        
        elif (self.reward == -1.0) and (self.mean_distance > self.need_range):
            
            for ant in self.hive_members.keys():
                new_cores = self.hive_members[ant].get_position() + np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)
        
        elif (self.reward == -1.0) and (self.mean_distance < self.need_range):

            for ant in self.hive_members.keys():
                new_cores = self.hive_members[ant].get_position() - np.random.normal(0.1, self.action_step_size)
                self.hive_members[ant].change_position(new_cores)
        
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

class STRATAGY_GENERATER(th.nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.layers = [
            th.nn.Linear(in_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, out_dim)
        ]
        self.main = th.nn.Sequential(*self.layers)
        self.onpolicy_reset()
        self.train()
    
    def onpolicy_reset(self):

        self.log_probs_list = []
        self.rewards_list = []
    
    def forward(self, state):
        
        self.pd_param = self.model(state)

    def act(self, state):

        self.state_net_output = th.from_numpy(state.astype(np.float32))
        self.forward(self.state_net_output)
        self.pd_distributed = th.distributions.Categorical(self.pd_param)
        self.action = self.pd_distributed.sample()
        self.log_pd_prob = self.pd_distributed.log_prob(self.action)
        self.log_probs_list.append(self.log_pd_prob)

        return self.action.item()




class REINGORCEMENT_ALGHRITM():

    def __init__(self, grad_learning_rate) -> None:
        
        self.grad_learning_rate = grad_learning_rate
        self.pi_stratagy = STRATAGY_GENERATER()
        self.optimizer = th.optim.Adam(self.pi_stratagy, lr=self.grad_learning_rate)


    def train(self):

        train_steps = len(self.pi_stratagy.rewards_list)
        rets = np.empty(train_step, )
        curent_ret = 0.0
        self.gamma = 0.99

        for train_step in train_steps:

            curent_ret = self.pi_stratagy.rewards_list[train_step] + self.gamma * curent_ret
            rets[train_step] = curent_ret
        
        rets_tensor = th.tensor(rets)
        log_probs_tensor = th.stack(self.pi_stratagy.log_probs_list)
        loss = -log_probs_tensor * rets_tensor
        loss_tensor = th.sum(loss)

        self.optimizer.zeros_grad()
        loss_tensor.backward()
        self.otpimizer.step()

        




if __name__ == "__main__":

    env = ENVIRONMENT(swarm_size=12, action_step_size=32.12)
    for t in range(200):
        state, reward = env.step()
        print(36*"=", state, reward, env.total_reward, 36*"=", sep="\n")

