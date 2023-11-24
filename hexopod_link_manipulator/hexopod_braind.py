import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")

figure, axis = plt.subplots()
axis_1 = axis.twinx()
axis_2 = axis.twinx()

class TD_LEARNING_MODEL():

    def __init__(self, normal_distrib_weight, learning_iterations, learning_rate) -> None:
        
        self.learning_iterations = learning_iterations
        self.learning_rate = learning_rate
        self.normal_distrib_weight = normal_distrib_weight
        self.param_info = pd.DataFrame(columns=["weights", "loss_lenght", "predicted_targets", "true_target"])
        self.weights_memory = np.zeros(100)
        self.fit_call = 0

    def fit(self, input_layer, target_layer):
        
        self.input_layer = input_layer
        self.target_layer = target_layer
        self.loss_vector = np.zeros(self.learning_iterations)
        self.weight_layer = np.zeros(self.input_layer.shape)

        if self.fit_call == 0:

            for iteration in range(self.learning_iterations):

                self.loss = 0
                self.predict = self.prediction()
                for (input_example_index, input_example) in enumerate(self.input_layer):

                    self.error_lenght = self.target_layer[input_example_index] - self.predict
                    self.copy_weight = self.weight_layer[input_example_index]
                    self.weight_layer[input_example_index] = self.copy_weight -  self.learning_rate * self.error_lenght
                    self.loss += self.error_lenght ** 2

                self.loss_vector[iteration] = self.loss

                self.param_info["weights"] = self.weight_layer
                self.param_info["loss_lenght"] = self.loss_vector
                self.param_info["predicted_targets"] = self.predict
                self.param_info["true_target"] = self.target_layer
        
        else:

            for iteration in range(self.learning_iterations):

                self.loss = 0
                self.predict = self.prediction()
                for (input_example_index, input_example) in enumerate(self.input_layer):

                    self.error_lenght = self.target_layer[input_example_index] - self.predict
                    self.copy_weight = self.weights_memory[input_example_index]
                    self.weights_memory[input_example_index] = self.copy_weight -  self.learning_rate * self.error_lenght
                    self.loss += self.error_lenght ** 2

                self.loss_vector[iteration] = self.loss

                self.param_info["weights"] = self.weights_memory
                self.param_info["loss_lenght"] = self.loss_vector
                self.param_info["predicted_targets"] = self.predict
                self.param_info["true_target"] = self.target_layer


        
        print(self.param_info)
        print(self.weights_memory)
        self.fit_call += 1

    def net_input(self):

        if self.fit_call == 0:
            return np.dot(self.input_layer, self.weight_layer)
        else:
            return np.dot(self.input_layer, self.weights_memory)
        
    def activation(self):

        net_input = self.net_input()
        neg_use_rate = np.random.randint(-34, 0)
        pos_use_rate = np.random.randint(0, 34)

        return np.where(net_input > 0, pos_use_rate, neg_use_rate)
    
    def prediction(self):

        target_prediction = self.activation()
        return target_prediction
    

x_data = np.random.normal(5.12, 5.6, 100)
Q_target_values = np.where(x_data > 6.0, np.random.randint(-38.5, 0), np.random.randint(0, 38.5))


td_learning_object = TD_LEARNING_MODEL(normal_distrib_weight=0.12,
                                       learning_iterations=100,
                                       learning_rate=0.01)
td_learning_object.fit(input_layer=x_data, target_layer=Q_target_values)
print(td_learning_object.weights_memory)
td_learning_object.fit(input_layer=x_data, target_layer=Q_target_values)
print(td_learning_object.weights_memory)




