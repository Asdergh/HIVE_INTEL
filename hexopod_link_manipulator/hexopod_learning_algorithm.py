import numpy as np
import pandas as pd
import tensorflow as tf
import torch as th
import matplotlib.pyplot as plt 
import mujoco_py as mj



class HEXOPOD_AGENT():
        
        def __init__(self) -> None:
                pass
        
        def step(self):
                pass

        def reset_state(self):
                pass






class Q_BASED_REINFORCE_ALGORITHM():

        def __init__(self, data_shape, epizodes_count, internel_count) -> None:
                
                self.sample_shape = data_shape[0]
                self.params_shape = data_shape[1]
                self.environment_class = HEXOPOD_AGENT()
                self.epizodes_count = epizodes_count
                self.steps = internel_count

                self.hyp_params = {
                        "td_net":
                        {
                                "first layer": {
                                        "output_shaspe:": 1,
                                        "activation": "relu",
                                        "input_shape": self.params_shape
                                },
                                "second layer": {
                                        "output_shape": 1,
                                        "activation": "relu"
                                },
                                "thred layer": {
                                        "output_shape": 1,
                                        "activation": "relu"
                                },
                                "fouyrth layer": {
                                        "output_shape": 1,
                                        "activation": "none"
                                },
                                "compile part": {
                                        "optimizer": "rmsprop",
                                        "loss_function": "mse",
                                        "accuracy_metrics": "mae"
                                },
                                "fit part": {
                                        "epochs": 100,
                                        "batch_size": 1,
                                        "verbose": 0
                                }
                        },
                        "state_to_action_net":
                        {
                                "first layer": {
                                        "ouput_shape": self.params_shape,
                                        "activation": "linear"
                                },
                                "second layer": {
                                        "output_shape": 7,
                                        "activation": "linear"
                                },
                                "thred layers": {
                                        "output_shape": 7,
                                        "activation": "linear"
                                },
                                "compile part": {
                                        "optimizer": "rmsprop",
                                        "loss_function": "mse",
                                        "accuracy_metrics": "mae"
                                },
                                "fit part": {
                                        "epochs": 100,
                                        "batch_size": 1,
                                        "verbose": 0
                                }
                        },

                        "general_network":
                        {
                                "learning_rate": 0.01,
                                "discont_coeff": 0.129,
                                "max_epizode": 100,
                                "learning_step": 100
                        }
                }

                self.td_model = tf.keras.Sequential()
                self.td_model.add(tf.keras.layers.Dense(self.hyp_params["td_model"]["first_layer"]["output_shape"], 
                                                     activation=self.hyp_params["td_model"]["first_layer"]["activation"],
                                                      input_shape=self.hyp_params["td_model"]["first_layer"]["input_shape"]))
                self.td_model.add(tf.keras.layers.Dense(self.hyp_params["td_model"]["second_layer"]["output_shape"], activation=self.hyp_params["td_model"]["activation"]))
                self.td_model.add(tf.keras.layers.Dense(self.hyp_params["td_model"]["thred_layer"]["output_shape"], activation=self.hyp_params["td_model"]["thred_layer"]["activation"]))
                self.td_model.add(tf.keras.layers.Dense(self.hyp_params["td_model"]["fourth_layer"]["output_shape"]))

                self.td_model.compile(optimizer=self.hyp_params["td_model"]["compile part"]["optimizer"],
                                   loss=self.hyp_params["td_model"]["compile part"]["loss_function"],
                                   accuracy=[self.hyp_params["td_model"]["compile"]["accuracy_metrics"]])
                
                self.state_to_action_model = tf.keras.Sequential()
                self.state_to_action_model.add(tf.keras.layers.Dense(self.hyp_params["state_to_action_net"]["first layer"]["ouput_shape"], 
                                                                     activation=self.hyp_params["state_to_action_net"]["first layer"]["activation"]),)
                self.state_to_action_model.add(tf.keras.layers.Dense(7, activation=self.hyp_params["state_to_action_net"]["second layer"]["activation"]))
                self.state_to_action_model.add(tf.keras.layers.Dense(7), activation=self.hyp_params["state_to_action_net"]["thred layer"]["activation"])

                self.state_to_action_model.compile(optimizer=self.hyp_params["state_to_action_net"]["compile part"]["optimizer"],
                                      loss=self.hyp_params["state_to_action_net"]["compile part"]["loss_function"],
                                      accuracy=[self.hyp_params["state_to_action_net"]["compile part"]["accuracy_metrics"]])
                

        
        def start_process(self):
                
                self.Q_randomize_score = np.random.randint(-100, 100, (100, ), dtype="float32")
                for _ in range(self.epizodes_count):
                        
                        self.state_list = []
                        self.reward_list = []
                        self.action_list = []

                        self.state = self.environment_class.reset_state()
                        self.state_list.append(self.state)

                        for _ in range(self.steps):
                                
                                self.net_output = self.state_to_action_model(self.state)
                                self.action_distribution = th.distributions.Normal(self.net_output)
                                self.action = self.action_distribution.sample()
                                self.state_list.append(self.environment_class.step(action=self.action))
                                self.reward_list.append(self.environment_class.reward)
                                self.action_list.append(self.action)
                        
                        self.train_data = 
                        
                        
                        
                               


                        


                        

                        







                
