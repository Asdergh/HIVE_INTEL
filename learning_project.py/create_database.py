import os
import cv2
import numpy as np
import pandas as pd



class DataMaker():

    def __init__(self, image_size, gray_mode, class_labels, samples_number_per_class, base_dir_name) -> None:
        
        self.image_size = image_size
        self.gray_mode = gray_mode
        self.class_labels = class_labels
        self.samples_number_per_class = samples_number_per_class
        self.base_dir_name = base_dir_name
        self.data_base_builded = False
        self.curent_frame_number = 0
    
    def generate_data(self):

        self.base_dir_name = os.path.join(self.base_dir_name, "data_dir")

        self.base_dir_train = os.path.join(self.base_dir_name, "train")
        self.base_dir_validation = os.path.join(self.base_dir_name, "validation")
        self.base_dir_test = os.path.join(self.base_dir_name, "test")

        os.mkdir(self.base_dir_name)
        os.mkdir(self.base_dir_train)
        os.mkdir(self.base_dir_validation)
        os.mkdir(self.base_dir_test)

        for class_label in self.class_labels:

            start_key = input("enter [Y] if you are ready to start loading data:\t")
            cap = cv2.VideoCapture(0)
            
            class_dir_train = os.path.join(self.base_dir_train, class_label)
            class_dir_test = os.path.join(self.base_dir_test, class_label)
            class_dir_validation = os.path.join(self.base_dir_validation, class_label)
            
            os.mkdir(class_dir_train)
            os.mkdir(class_dir_test)
            os.mkdir(class_dir_validation)

            while True:
                
                if (start_key == "Y" or start_key == "y"):
                    
                    print("test inf")
                    for self.curent_frame_number in range(self.samples_number_per_class):
                        
                        _, frame = cap.read()
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        cv2.imwrite(f"{class_dir_train}\\{class_label}_{self.curent_frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN TRAIN DIR: [{class_dir_train}\\{class_label}_{self.curent_frame_number}.jpg]")
                    
                    for self.curent_frame_number in range(self.samples_number_per_class):

                        _, frame = cap.read()
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        cv2.imwrite(f"{class_dir_validation}\\{class_label}_{self.curent_frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN VALIDATION DIR: [{class_dir_validation}\\{class_label}_{self.curent_frame_number}.jpg]")
                    
                    for self.curent_frame_number in range(self.samples_number_per_class):

                        _, frame = cap.read()
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        cv2.imwrite(f"{class_dir_test}\\{class_label}_{self.curent_frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN TEST DIR: [{class_dir_test}\\{class_label}_{self.curent_frame_number}.jpg]") 

                    break
                
                else:
                    pass
                



data_generator_object = DataMaker(
    image_size=(214, 214),
    gray_mode=True,
    class_labels=["hand", "face", "lamp"],
    samples_number_per_class=200,
    base_dir_name="C:\\Users\\1\\Desktop\\reinforcment_deep_ML"
)

#data_generator_object.generate_data()      


