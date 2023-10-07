import numpy as np
import pandas as pd
from hive_file_descriptor import COPTER_HIVE_MODULE_FILE_DESKRIPTOR



class ANT_HIVE_MEMBER_DYNAMIC_CALC():

    def __init__(self, time_iteration_count) -> None:
        
        self.file_descriptor = COPTER_HIVE_MODULE_FILE_DESKRIPTOR()

        self.ant_anguler_velocity_list = []
        self.ant_anguler_acceleration_list = []
        self.ant_anguler_velocity_cmd_list = []
        self.ant_magnetic_dockers_mode_list = []

        self.ant_velocity_list = []
        self.ant_position_list = []
        self.ant_acceleration_list = []
        self.ant_anguler_vel_list = []
        self.ant_anguler_acc_list = []

        self.curent_position_vector = np.zeros(3)
        self.curent_velocity_vector = np.zeros(3)
        self.curent_acceleration_vector = np.zeros(3)

        self.integrated_position_vector = np.zeros(3)
        self.integrated_velocity_vector = np.zeros(3)
        self.integrated_anguler_velocity_vector = np.zeros(3)
        self.integrated_psi_angle = 0
        self.integrated_theta_angle = 0
        self.integrated_sigma_angle = 0

        self.curent_anguler_velocity_vector = np.zeros(3)
        self.curent_anguler_acceleration_vector = np.zeros(3)
        self.curent_anguler_vel_cmd = np.zeros(4)
        
        
        self.psi = 0
        self.theta = 0
        self.sigma = 0

        self.anguler_velocity_cmd_vector = [0.0, 0.0, 0.0, 0.0]

        self.ant_thurst_coeef = 0.65
        self.ant_d_coeff = 5.678
        self.ant_link_lenght = 200
        

        self.rotation_matrix_3D = np.array([
            [np.cos(self.sigma) * np.cos(self.theta), np.cos(self.sigma) * np.sin(self.theta) * np.sin(self.psi) - np.sin(self.sigma) * np.cos(self.psi), 
             np.cos(self.sigma) * np.sin(self.theta) * np.cos(self.psi) + np.sin(self.sigma) * np.sin(self.sigma)],
            [np.sin(self.sigma) * np.cos(self.theta), np.sin(self.sigma) * np.sin(self.theta) * np.sin(self.psi) + np.cos(self.sigma) * np.cos(self.psi), 
              np.sin(self.sigma) * np.sin(self.theta) * np.cos(self.psi) - np.cos(self.sigma) * np.sin(self.psi)],
            [-np.sin(self.theta), np.cos(self.theta) * np.sin(self.sigma), np.cos(self.theta) * np.cos(self.sigma)]
        ])

        self.rotation_matrix_2D = np.array([
            [np.cos(self.psi), np.sin(self.psi)],
            [-np.sin(self.theta), np.cos(self.theta)]
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

        self.normolize_vector = np.array([0.0, 0.0, 1.0])
    
    def calculate_curent_acceleration(self):

        self.curent_acceleration_vector = self.rotation_matrix_3D.dot(sum(self.anguler_velocity_cmd_vector) *self.ant_thurst_coeff / self.ant_mass * self.normolize)
        
        self.tensor_of_inertia = np.array([
            [5.828570, 0, 0],
            [0, 7.169112, 0],
            [0, 0, 0.000001]
        ])
        self.thrust_moment = np.array([
            self.ant_link_lenght * self.ant_d_coeff * (self.anguler_velocity_cmd_vector[0] ** 2 - self.anguler_velocity_cmd_vector[1] ** 2),
            self.ant_link_lenght * self.ant_d_coeff * (self.anguler_velocity_cmd_vector[3] ** 2 - self.anguler_velocity_cmd_vector[1] ** 2),
            self.ant_link_lenght * self.d_coeff * (self.anguler_velocity_cmd_vector[3] ** 2 + self.anguler_velocity_cmd_vector[1] ** 2 - self.anguler_velocity_cmd_vector[0] ** 2 - self.anguler_velocity_cmd_vector[2] ** 2)
        ])

        self.ant_anguler_acceleration_list = (self.thrust_moment - np.cross(self.curent_anguler_velocity_vector, self.tensor_of_inertia.dot(self.curent_anguler_velocity_vector)))\
            .dot(np.linalg.inv(self.tensor_of_inertia))
    
    def general_claculations(self):

        self.integrated_velocity_vector += self.curent_acceleration_vector * 0.01
        self.integrated_position_vector += self.integrated_velocity_vector * 0.01
        self.integrated_anguler_velocity_vector += self.curent_anguler_acceleration_vector * 0.01
        self.integrated_psi_angle += self.integrated_anguler_velocity_vector[0] * 0.01
        self.integrated_theta_angle += self.integrated_anguler_velocity_vector[1] * 0.01
        self.integrated_sigma_angle += self.integrated_anguler_velocity_vector[2] * 0.01
        
class HIVE_CONTROL_SISTEM(ANT_HIVE_MEMBER_DYNAMIC_CALC):

    def __init__(self, time_iteration_count) -> None:
        super().__init__(time_iteration_count)

        self.K_p = 0.0
        self.K_d = 0.0
        self.K_i = 0.0

        
        self.pid_desired_anguler_vel_vector = np.zeros(3)
        self.pid_desired_anguler_acc_vector = np.zeros(3)

        self.pid_desired_psi = 0
        self.pid_desired_theta = 0
        self.pid_desired_sigma = 0

        self.pid_desired_anguler_acc_past_vector = np.zeros(3)
        self.pid_desired_anguler_vel_past_vector = np.zeros(3)
        self.pid_position_past_vector = np.zeros(3)      
        
        self.pid_desired_psi_past = 0
        self.pid_desired_theta_past = 0
        self.pid_desired_sigma_past = 0

        self.position_integr_x = 0
        self.position_integr_y = 0
        self.position_integr_z = 0

        self.anguler_vel_integr = np.zeros(3)

        self.psi_integr = 0
        self.theta_integr = 0
        self.sigma_integr = 0


    def calculate_desired_angles(self):

        self.position_x_error = self.pid_desired_psition_vector[0] - self.curent_position_vector[0]
        self.position_y_error = self.curent_position_vector[1] - self.pid_desired_position_vector[1]
        self.position_z_error = self.curent_position_vector[2] - self.pid_desired_position_vector[2]

        self.position_integr_x += self.position_x_error * 0.01
        self.position_integr_y += self.position_y_error * 0.01
        self.position_integr_z += self.position_z_error * 0.01

        self.pid_desired_psi = self.K_p * self.position_x_error + self.K_i * self.position_integr_x + self.K_d * (self.curent_position_vector[0] - self.pid_position_past_vector[0]) / 0.01
        self.pid_desired_theta = self.K_p * self.position_y_error + self.K_i * self.position_integr_y + self.K_d * (self.curent_position_vector[1] - self.pid_position_past_vector[1]) / 0.01
        

        self.pid_position_past_vector[0] = self.curent_position_vector[0]
        self.pid_position_past_vector[1] = self.curent_position_vector[1]
        self.pid_position_past_vector[2] = self.curent_position_vector[2]
    
    def calculate_desired_anguler_vel(self):

        self.pid_psi_error = self.pid_desired_psi - self.integrated_psi_angle
        self.pid_theta_error = self.pid_desired_theta - self.integrated_theta_angle
        self.pid_sigma_error = self.pid_desired_sigma - self.integrated_sigma_angle

        self.psi_integr += self.pid_psi_error * 0.01
        self.theta_integr += self.pid_theta_error * 0.01
        self.sigma_integr += self.pid_sigma_error * 0.01

        self.pid_desired_anguler_vel_vector[0] = self.K_p * self.pid_psi_error + self.K_i * self.psi_integr + self.K_d * (self.psi_integr - self.pid_desired_psi_past) / 0.01
        self.pid_desired_anguler_vel_vector[1] = self.K_p * self.pid_theta_error + self.K_i * self.theta_integr + self.K_d * (self.theta_integr - self.pid_desired_theta_past) / 0.01
        self.pid_desired_anguler_vel_vector[2] = self.K_p * self.pid_sigma_error + self.K_i * self.sigma_integr + self.K_d * (self.sigma_integr - self.pid_desired_sigma_past) / 0.01

        self.pid_desired_psi_past = self.pid_desired_psi
        self.pid_desired_theta_past = self.pid_desired_theta
        self.pid_desired_sigma_past = self.pid_desired_sigma
    
    def calculate_desired_anguler_acc(self):

        self.pid_anguler_vel_error = self.pid_desired_anguler_vel_vector - self.integrated_anguler_velocity_vector
        self.anguler_vel_integr += self.pid_anguler_vel_error * 0.01
        self.pid_desired_anguler_acc_vector = self.K_p * self.pid_anguler_vel_error + self.K_i * self.anguler_vel_integr + self.K_d * (self.pid_desired_anguler_vel_vector - self.pid_desired_anguler_vel_past_vector) / 0.01
        self.pid_desired_anguler_vel_past_vector = self.pid_desired_anguler_vel_vector
    
    def claculate_anguler_vel_cmd(self):
        # TODO: !!дописать миксер команд!!
        pass






    
    





    



        

        


