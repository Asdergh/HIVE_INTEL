import numpy as np
import pandas as pd
import json as js
import yaml as ym
import xml.etree.ElementTree as ET


class HIVE_FILE_DESCRIPTOR():

    def __init__(self, file_type, serialize_mode) -> None:
        
        self.file_type = file_type
        self.serialize_mode = serialize_mode
    
    def write_log(self, hive_members_info_dict):

        if self.file_type == ".json":
            with open(f"info_log{self.file_type}", "w") as json_file:
                js.dump(hive_members_info_dict, json_file)
        
        elif self.file_type == ".yaml":
            with open(f"info_log{self.file_type}", "w") as yaml_file:
                ym.dump(hive_members_info_dict, yaml_file)
        
        elif self.file_type == ".xml":
            
                



        