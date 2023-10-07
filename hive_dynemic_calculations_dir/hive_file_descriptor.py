import json as js
import numpy as np
import pandas as pd
import json as js
import xml
import yaml



class COPTER_HIVE_MODULE_FILE_DESKRIPTOR():

    def __init__(self) -> None:
        
        self.hive_members_info_dict = {
            "queen_drone_info": 
            {
                "speed": None,
                "position x": None,
                "position y": None,
                "potision z": None,
                "magnetic_dockers_mode_list": None,
                "anguler_velocity_commands_list": None
            }
        }
    
    def add_member_info(self, bluetooth_id, speed_list, position_list, magnetic_mode_list):

        speed_list = list(speed_list)
        position_list_x = list(position_list[:, 0])
        position_list_y = list(position_list[:, 1])
        position_list_z = list(position_list[:, 2])
        magnetic_mode_list = list(magnetic_mode_list)

        self.hive_members_info_dict[f"ant|id: {bluetooth_id}"]
        self.hive_members_info_dict[f"ant|id: {bluetooth_id}"] = {
            "speed": speed_list,
            "position x": position_list_x,
            "position y": position_list_y,
            "position z": position_list_z,
            "magnetic_dockers_mode_list": magnetic_mode_list,
        }
    
    def change_queen_info(self, position_list, velocity_list, anguler_velocity_cmd_list, magnetic_docker_list):

        self.queen_stats = self.hive_members_info_dict["queen_drone_info"]
        self.queen_stats["speed"] = velocity_list
        self.queen_stats["position x"] = position_list[:, 0]
        self.queen_stats["position y"] = position_list[:, 1]
        self.queen_stats["position z"] = position_list[:, 2]
        self.queen_stats["anguler_velocity_commands_list"] = anguler_velocity_cmd_list
        self.queen_stats["magnetic_dockers_mode_list"] = magnetic_docker_list
    
    def change_ant_info(self, position_list, velocity_list, anguler_velocity_cmd_list, magnetic_docker_list, bluetooth_id):

        self.ant_stats = self.hive_members_info_dict[f"ant|id: {bluetooth_id}"]
        self.ant_stats["speed"] = velocity_list
        self.ant_stats["position x"] = position_list[:, 0]
        self.ant_stats["position y"] = position_list[:, 1]
        self.ant_stats["position z"] = position_list[:, 2]
        self.ant_stats["anguler_velocity_commands_list"] = anguler_velocity_cmd_list
        self.ant_stats["magnetic_dockers_mode_list"] = magnetic_docker_list
    
    def save_data_states(self):

        with open("log_dir/hive_members_logs.json", "w") as json_file:
            js.dump(self.hive_members_info_dict, json_file)
        
        with open("log_dir/hive_members_logs.yaml", "w") as yaml_file:
            yaml.dump(self.hive_members_info_dict, json_file)
    
    
        



