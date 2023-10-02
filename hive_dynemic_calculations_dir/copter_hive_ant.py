import numpy as np
import pandas as pd
from copter_hive_file_desctiptor import COPTER_HIVE_MODULE_FILE_DESKRIPTOR



class ANT_HIVE_MEMBER_DYNAMIC_CALC():

    def __init__(self, time_iteration_count) -> None:
        
        self.ant_anguler_velocity_list = []
        self.ant_anguler_velocity_cmd_list = []
        self.ant_magnetic_dockers_mode_list = []

        self.ant_velocity_list = []
        self.ant_position_list = []
        self.ant_acceleration_list = []
        self.ant_anguler_vel_list = []
        self.ant_anguler_acc_list = []

        self.curent_position_vector = np.array([0.0, 0.0, 0.0])
        self.curent_velocity_vector = np.array([0.0, 0.0, 0.0])
        self.curent_acceleration_vector = np.array([0.0, 0.0, 0.0])
        self.curent_anguler_velocity_vector = np.array([0.0, 0.0, 0.0])
        self.curent_anguler_acceleration_vector = np.array([0.0, 0.0, 0.0])
        self.curent_anguler_vel_cmd = np.array([0.0, 0.0, 0.0, 0.0])
        self.curent_angle_vector = np.array([0.0, 0.0, 0.0])
        self.theta = self.curent_angle_vector[0]
        self.gamma = self.curent_angle_vector[1]
        self.sigma = self.curent_angle_vector[2]

        self.rotation_matrix_3D = np.array([
            [np.cos(self.curent_angle_vector[0]) * np.cos(self.curent_angle_vector[1] - np.sin(self.curent_angle_vector[0]))]
        ])

        self.time_iteration = 0
        self.time_iteration_count = time_iteration_count
    
    def calculate_path_trajectory(self, desired_position, desired_acceleration, desired_velocity):

        self.Time_matrix = np.array([
            [0, 0, 0, 0, 0, 1],
            [self.time_iteration ** 5, self.time_iteration ** 4, self.time_iteration ** 3, self.time_iteration ** 2, self.time_iteration, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * self.time_iteration ** 4, 4 * self.time_iteration ** 3, 3 * self.time_iteration ** 2, 2 * self.time_iteration, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * self.time_iteration ** 3, 12 * self.time_iteration ** 2, 6 * self.time_iteration, 2, 0, 0]
        ])

        self.ant_condition_matrix = np.array([
            [self.curent_position_vector[0], desired_position[0], self.curent_velocity_vector[0], desired_velocity[0], self.curent_acceleration_vector[0], desired_acceleration[0]],
            [self.curent_position_vector[1], desired_position[1], self.curent_velocity_vector[1], desired_velocity[1], self.curent_acceleration_vector[1], desired_acceleration[1]],
            [self.curent_position_vector[2], desired_position[2], self.curent_velocity_vector[2], desired_velocity[2], self.curent_acceleration_vector[2], desired_acceleration[2]]
        ])

        self.x_coeff = np.linalg.inv(self.Time_matrix).dot(self.ant_condition_matrix[0])
        self.y_coeff = np.linalg.inv(self.Time_matrix).dot(self.ant_condition_matrix[1])
        self.z_coeff = np.linalg.inv(self.Time_matrix).dot(self.ant_condition_matrix[2])

        self.curent_position_vector[0] = self.Time_matrix[1].dot(self.x_coeff)
        self.curent_position_vector[1] = self.Time_matrix[1].dot(self.y_coeff)
        self.curent_position_vector[2] = self.Time_matrix[1].dot(self.z_coeff)

        self.curent_velocity_vector[0] = self.Time_matrix[3].dot(self.x_coeff)
        self.curent_velocity_vector[1] = self.Time_matrix[3].dot(self.y_coeff)
        self.curent_velocity_vector[2] = self.Time_matrix[3].dot(self.z_coeff)

        self.curent_acceleration_vector[0] = self.Time_matrix[5].dot(self.x_coeff)
        self.curent_acceleration_vector[1] = self.Time_matrix[5].dot(self.y_coeff)
        self.curent_acceleration_vector[2] = self.Time_matrix[5].dot(self.z_coeff)


    



        

        


