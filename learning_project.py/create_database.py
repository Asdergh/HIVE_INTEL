import os
import cv2
import numpy as np
import pandas as pd



class DataMaker():

    def __init__(self, image_size, gray_mode, class_labels, samples_number, base_dir_name) -> None:
        
        self.image_size = image_size
        self.gray_mode = gray_mode
        self.class_labels = class_labels
        self.samples_number = samples_number
        self.base_dir_name = base_dir_name
        self.data_base_builded = False
    
    def generate_data(self):

        for class_label in self.class_labels:

            class_samples_train = os.path.join(self.base_dir_name, f"{class_label}_train")
            class_samples_validation = os.path.join(self.base_dir_name, f"{class_label}_validation")
            class_samples_test = os.path.join(self.base_dir_name, f"{class_label}_test")

            os.mkdir(class_samples_train)
            os.mkdir(class_samples_test)
            os.mkdir(class_samples_validation)

            start_key = input("enter [Y] if you are ready to start loading data:\t")
            cap = cv2.VideoCapture(0)
            
            while True:

                if (start_key == "Y" or start_key == "y"):
                    for frame_number in range(self.samples_number):
                        _, frame = cap.read()
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(f"{class_samples_train}\\{frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN FILE TRAIN: [{class_samples_train}\\{frame_number}.jpg]")

                    for frame_number in range(self.samples_number):
                        _, frame = cap.read()
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(f"{class_samples_validation}\\{frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN FILE VALIDATION: [{class_samples_train}\\{frame_number}.jpg]")

                    for frame_number in range(self.samples_number):
                        _, frame = cap.read()
                        frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))
                        if self.gray_mode == True:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(f"{class_samples_test}\\{frame_number}.jpg", frame)
                        print(f"IMAGE WAS WRITEN IN FILE TEST: [{class_samples_train}\\{frame_number}.jpg]")

                    break

                else:
                    break
        
        return self.data_base_builded


data_generator_object = DataMaker(
    image_size=(214, 214),
    gray_mode=True,
    class_labels=["hand", "face", "lamp"],
    samples_number=200,
    base_dir_name="C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\test_data_dir"
)

data_generator_object.generate_data()

                        


