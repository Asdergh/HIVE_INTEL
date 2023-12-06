import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os

# COLLECTING PATHS OF IMAGES
image_directory = "C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\cats_and_dogs_dataset"
one_image_to_analize = "C:\\Users\\1\\Desktop\\reinforcment_deep_ML\\cats_and_dogs_dataset\\train\\cats\\567.jpg"

train_images_path = os.path.join(image_directory, "train")
validation_images_path = os.path.join(image_directory, "validation")
test_images_path =  os.path.join(image_directory, "test")

# MODEL BUILDING
model = tf.keras.Sequential()
# LAYER CONV1 MOUDULE1
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# LAYER CONV2 MODULE2
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
# LAYER CONV3 MODULE3
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# LAYER CONV4 MODULE4
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# LAYER CONV5 MODULE5
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# LAYER FLATTEN1 MODULE6
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(568, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# MODEL COMPILE
model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
)

# INIT IMAGE GENERATOR

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255.,
    rotation_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2
)
image_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

# PREPEARE IMAGE GENERATOR
train_generator = image_datagen.flow_from_directory(
    train_images_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)
validation_generator = image_datagen.flow_from_directory(
    validation_images_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)
test_generator = image_datagen.flow_from_directory(
    test_images_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

"""history = model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=20
)
"""
image = tf.keras.preprocessing.image.load_img(one_image_to_analize, target_size=(150, 150))
image_tensor = tf.keras.preprocessing.image.img_to_array(image)
image_tensor = np.expand_dims(image_tensor, axis=0)
image_tensor /= 255.

print(image_tensor.shape)
