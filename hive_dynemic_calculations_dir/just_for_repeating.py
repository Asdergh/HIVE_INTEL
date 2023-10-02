import numpy as np
import torch as th
import gym as gm

gamma = 0.99
class PI(th.nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = [
            th.nn.Linear(in_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, out_dim)
        ]
        self.model = th.nn.Sequential()
        self.onpolicy_reset(у)
        self.train()
    
    def onpolicy_reset(self):

        self.rewards_list = []
        self.log_probs_list = []
    
    def net_forward(self, x):
        self.pd_param = self.model(x)
    
    def act(self, state):

        self.out_net_item = th.from_numpy(state.astype(np.float32))
        self.net_forward(self.out_net_item)
        self.pd_distrib_param = th.distributions.Categorical(self.pd_param)
        self.action = self.pd_distrib_param.sample()
        self.log_prob = self.pd_distrib_param.log_prob(self.action)
        self.log_probs_list.append(self.log_prob)
        
        return self.action.item()

    def train(self)

