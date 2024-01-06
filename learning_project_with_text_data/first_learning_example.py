import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_sequences = 10000
max_words = 10000

input_layer = tf.keras.Input(shape=(None, max_sequences, max_words))
conv1D_model_layer = tf.keras.layers.Conv1D(32, 5, activation="relu")(input_layer)
conv1D_model_layer = tf.keras.layers.MaxPooling1D(5)(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.Conv1D(64, 5, activation="relu")(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.Conv1D(64, 5, activation="relu")(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.MaxPooling1D(5)(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.Conv1D(128, 5, activation="relu")
conv1D_model_layer = tf.keras.layers.Conv1D(128, 5, activation="relu")
conv1D_model_layer = tf.keras.layers.MaxPooling1D(5)
conv1D_model_layer = tf.keras.layers.Conv1D(128, 5, activation="relu")(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(conv1D_model_layer)
conv1D_model_layer = tf.keras.layers.Dense(32, activation="relu")

predicted_class_output = tf.keras.layers.Dense(10, activation="sigmoid", name="predicted class")(conv1D_model_layer)
predicted_volume_output = tf.keras.layers.Dense(1, name="predicted value")(conv1D_model_layer)

conv_model = tf.keras.Model(input_layer, [predicted_class_output, predicted_volume_output])
conv_model.compile(
    optimizer="rmsprop",
    loss={
        "predicted class": "categorical_crossentropy",
        "predicted value": "mse"
    },
    metrics={
        "predicted class": "acc",
        "predicted value": "mae"
    }
)

conv_model.compile
