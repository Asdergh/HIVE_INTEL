import numpy as np
import torch as th
import random as rd



class PI(th.nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = [
            th.nn.Linear(in_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, out_dim)
        ]
        self.model = th.nn.Sequential(*self.layers)
        self.onpolicy_reset()
        self.train()
        self.gamma = 0.99
    
    def onpolicy_reset(self):

        self.log_probs_list = []
        self.rewards_list = []
    
    def net_out(self, state):
        
        self.pd_param = self.model(state)
    
    def act(self, state):

        self.state = th.from_numpy(state.astype(np.float32))
        self.net_out(self.state)
        self.pd = th.distributions.Categorical(self.pd_param)
        self.action = self.pd.sample()
        self.log_prob = self.pd.log_prob(self.action)
        self.log_probs_list.append(self.log_prob)

        return self.action.item()


    