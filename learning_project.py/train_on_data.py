import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from create_database import DataMaker



class ObjectDetector():

    def __init__(self) -> None:
        
        
        self.data_maker = DataMaker(
            image_size=(214, 214),
            gray_mode=False,
            class_labels=["hand", "face", "lamp"],
            samples_number_per_class=200,
            base_dir_name="C:\\Users\\1\\Desktop\\reinforcment_deep_ML"
            )
        
        self.data_maker.generate_data()
        
        self.model_dir = "C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\detector_model.h5"

        self.conv_model = tf.keras.Sequential()
        self.conv_model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(self.data_maker.image_size[0], self.data_maker.image_size[1], 3), activation="relu"))
        self.conv_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.conv_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.conv_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.conv_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.conv_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.conv_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        self.conv_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.conv_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        self.conv_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.conv_model.add(tf.keras.layers.Flatten())
        self.conv_model.add(tf.keras.layers.Dense(512, activation="relu"))
        self.conv_model.add(tf.keras.layers.Dense(len(self.data_maker.class_labels), activation="linear"))

        self.conv_model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["acc"]
        )
        self.conv_model.summary()
    
    def train_process(self):

        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2
        )

        self.train_datagen = self.image_generator.flow_from_directory(
            self.data_maker.base_dir_train,
            target_size=(214, 214),
            batch_size=20,
            class_mode="categorical"
        )
        self.test_datagen = self.image_generator.flow_from_directory(
            self.data_maker.base_dir_test,
            target_size=(214, 214),
            batch_size=20,
            class_mode="categorical"
        )
        self.validation_datagen = self.image_generator.flow_from_directory(
            self.data_maker.base_dir_validation,
            target_size=(214, 214),
            batch_size=20,
            class_mode="categorical"
        )

        self.train_history = self.conv_model.fit_generator(
            self.train_datagen,
            steps_per_epoch=10,
            epochs=3,
            validation_data=self.validation_datagen,
            validation_steps=10
        )
        self.conv_model.save(self.model_dir)


    def plot_losses_and_accuraces(self):

        fig, axis = plt.subplots(nrows=2)
        losses = [np.asarray(self.train_history.history["loss"]), np.asarray(self.train_history.history["val_loss"])]
        accuraces = [np.asarray(self.train_history.history["acc"]), np.asarray(self.train_history.history["val_acc"])]
        colors = ["blue", "orange"]
        labels = [["loss info", "acc info"], ["val loss info", "val acc info"]]

        for (loss, acc, color, label) in zip(losses, accuraces, colors, labels):
            
            axis[0].plot(range(1, loss.shape[0] + 1), loss, color=color, label=label[0])
            axis[0].fill_between(range(1, loss.shape[0] + 1), loss - 0.12, loss + 0.12, color=color, alpha=0.34)

            axis[1].plot(range(1, acc.shape[0] + 1), acc, color=color, label=label[1])
            axis[1].fill_between(range(1, acc.shape[0] + 1), acc - 0.12, acc + 0.12, color=color, alpha=0.34)
        
        axis[0].legend(loc="upper left")
        axis[1].legend(loc="upper left")
        plt.show()

    
    def object_detection(self):
        
        cap = cv2.VideoCapture(0)

        while True:

            _, frame = cap.read()
            frame = np.asarray(frame)
            frame = np.expand_dims(frame, axis=1)

            prediction = self.conv_model.predict(frame)
            curent_item_detected = self.conv_model[:np.argmax(prediction)]
            last_conv_layer = self.conv_model.get_layer("")


        

ODM = ObjectDetector()

ODM.train_process()
ODM.plot_losses_and_accuraces()
ODM.object_detection()

        