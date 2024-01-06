import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class ImageDetectionNN():

    def __init__(self, layers_depth=4, conf_filters_size=(3, 3),
                 max_pooling_mode=True, branchs_count=3, model_image_size=(224, 224),
                 filters_count=32, pooling_size=(2, 2)) -> None:
        
         self.layers_depth = layers_depth
         self.conv_filters_size = conf_filters_size
         self.max_pooling_mode = max_pooling_mode
         self.branchs_count = branchs_count
         self.model_image_size = model_image_size
         self.filters_count = filters_count
         self.poolint_size = pooling_size
    
    def build_model(self):

        input_tensor = tf.keras.Input(shape=(224, 224, 3))
        self.branchs = {}
        for branch in range(self.branchs_count):
            layers_stack = []
            for layer in range(self.layers_depth):
                
                if layer == 1:

                    if self.max_pooling_mode == True:
                        layers_stack.append(tf.keras.layers.Conv2D(self.filters_count, self.conv_filters_size, activation="relu")(input_tensor))
                        layers_stack.append(tf.keras.layers.MaxPooling2D(self.pooling_size)(layers_stack[-1]))
                    else:
                        layers_stack.append(tf.keras.layes.Conv2D(self.filters_count, self.conv_filters_size, activation="relu")(input_tensor))
                else:

                    if self.max_pooling_mode == True:
                        layers_stack.append(tf.keras.layers.Conv2D(self.filters_count, self.conv_filters_size, activation="relu")(layers_stack[-1]))
                        layers_stack.append(tf.keras.layers.MaxPooling2D(self.pooling_size)(layers_stack[-1]))
                    else:
                        layers_stack.append(tf.keras.layes.Conv2D(self.filters_count, self.conv_filters_size, activation="relu")(layers_stack[-1]))
                
                if layer == self.layers_depth:
                    layers_stack.append(tf.keras.layers.Flatten()(layers_stack[-1]))

            self.branchs[f"branch{branch}"] = layers_stack

        result_layer = tf.keras.layers.concatenate(self.branchs["branch1"][-1],
                                                   self.branchs["branch2"][-1],
                                                   self.branchs["branch3"][-1], axis=-1)
        classification_layer = tf.keras.layers.Dense(32, activation="relu")(result_layer)
        classification_layer = tf.keras.layers.Dense(1, activation="sigmoid")(classification_layer)
        
        self.model = tf.keras.Model(input_tensor, classification_layer)
        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics="accuracy"
        )
        
        
                    

