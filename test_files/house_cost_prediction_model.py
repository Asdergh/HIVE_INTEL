import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#=======================================================================================================================================================================
#ENVIRONMENT CLASS| SET AND GET ACTIONS, REWARD AND STATE
#=======================================================================================================================================================================
class Environment():

    def __init__(self) -> None:
        
        
        self.state_vector = np.zeros(3)
        self.rewards_list = []
        self.total_reward = 0
        self.target_point = np.array([5.678, 3.456, 9.123])
        self.past_state = np.zeros(3)
        self.reward = 0
    
    def step(self, action_vector):
        
        if (np.sqrt(self.state_vector[0] - self.target_point[0]) ** 2 + (self.state_vector[1] + self.target_point[1]) ** 2 + (self.state_vector[2] - self.target_point[2]) ** 2) > \
        (np.sqrt((self.past_state[0] - self.target_point[0]) ** 2 + (self.past_state[1] - self.target_point[1]) ** 2 + (self.past_state[2] - self.target_point[2]) ** 2)):
             self.reward = np.random.normal(0.09, 1.0)
        else:
             self.reward = -np.random.normal(0.09, 1.0)
        
        self.state_vector = action_vector
        self.iteration += 1
        self.past_state = self.state_vector
        return self.state_vector, self.reward

         

# ==================================================================================================================================================================
#ACTION CHOICE ALGORITHM | LEARNING PROCESS ALGORITHM
# ================================================================================================================================================================== 
class ActionChoiceAlgorithm(Environment):
        
        def __init__(self) -> None: 
             
            super().__init__()

            self.learning_iterations = 100
            self.episode_count = 100
            self.sample_count = 100

            self.rewards_list = []
            self.states_list = []
            self.actions_list = []

            self.total_reward = 0
            self.grade_sa = np.random.normal(0.09, 1.09, self.sample_count)
            self.past_grade_sa = self.grade_sa

            self.action_model = tf.keras.Sequential()
            self.action_model.add(tf.keras.layers.Dense(3, activation="relu", input_shape=(3, )))
            self.action_model.add(tf.keras.layers.Dense(3, activation="relu"))
            self.action_model.add(tf.keras.layers.Dense(3))

            self.grade_model = tf.keras.Sequential()
            self.grade_model.add(tf.keras.layers.Dense(7, input_shape=(7, ), activation="relu"))
            self.grade_model.add(tf.keras.layers.Dense(7, activation="relu"))
            self.grade_model.add(tf.keras.layers.Dense(7, activation="relu"))
            self.grade_model.add(tf.keras.layers.Dense(1))
            self.grade_model.compile(
                 optimizer="emsprop",
                 loss="categorical_crossentropy",
                 metrics=["accuracy"]
            )

            self.loss_function = np.zeros(shape=self.grade_model.shape)



        def train_process(self):
             
            # main learning cycle
            for epizode in range(self.episode_count):
                
                # cycle for updating state, action and reward
                action = np.random.randint(0.09, 1.09, 3)
                for sample_index in range(self.sample_count):
                       
                    # get reward, state and action
                    state, reward = self.step(action)
                    action = self.action_model(state)
                    self.action_list.append(action)
                    self.states_list.append(state)
                    self.rewards_list.append(reward)
                
                # make tensor from action, state and reward lists
                tensor_of_action = np.asarray(self.actions_list)
                tensor_of_state = np.asarray(self.states_list)
                tensor_of_reward = np.asarray(self.rewards_list)

                # standartaze data tensors
                std_train_action = (tensor_of_action - np.mean(tensor_of_action)) / (tensor_of_action - np.mean(tensor_of_action)) ** 2
                std_train_state = (tensor_of_state - np.mean(tensor_of_state)) / (tensor_of_state - np.mena(tensor_of_state)) ** 2
                std_train_reward = (tensor_of_reward - np.mena(tensor_of_reward)) / (tensor_of_reward - np.mean(tensor_of_reward)) ** 2

                # making main train tensor
                train_tensor = np.concatenate(std_train_action,
                                              std_train_state, 
                                              std_train_reward, axis=1)
                # cycle for TD learning
                for learning_iteration in range(self.learnign_iterations):
                    
                    self.grade_history = self.grade_model.fit((train_tensor, self.past_grade_sa))
                    loss = self.grade_history - self.past_grade_sa
                    self.loss_function.append(loss)
                    self.past_grade_sa = self.grade_model(train_tensor)
                
                result_data = np.concatenate(train_tensor,
                                             self.grade_model(train_tensor), axis=1)
                self.Gr_result_table = pd.DataFrame(data=result_data)

            print(self.Gr_result_table)
            return self.Gr_result_table, self.loss_function


learning_sample = ActionChoiceAlgorithm()
Q_table, loss = learning_sample.train_process()

                



                     
                

                
                
                

                       



