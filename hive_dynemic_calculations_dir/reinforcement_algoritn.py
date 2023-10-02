import torch as th
import numpy as np
import pandas as pd
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
        self.model = th.nn.Sequential(*self.layers)
        self.onpolicy_reset()
        self.train()
    
    def onpolicy_reset(self):

        self.log_probs_list = []
        self.rewards_list = []
    
    def forward_throught_model(self, x):
        self.pd_param = self.model(x)

    def act(self, state):

        self.x = th.from_numpy(state.astype(np.float32))
        self.forward_throught_model(self.x)
        self.pd = th.distributions.Categorical(logits=self.pd_param)
        self.action = self.pd.sample()
        self.log_prob = self.pd.log_prob(self.action)
        self.log_probs_list.append(self.log_prob)

        return self.action.item()


def train(pi, optimizer):
    
    T = len(pi.rewards_list)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in range(T):
        future_ret = pi.rewards_list[t] + gamma * future_ret
        rets[t] = future_ret
    
    rets = th.tensor(rets)
    log_probs = th.stack(pi.log_probs_list)
    loss = -log_probs * rets
    loss = th.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def main():
    
    env = gm.make("CartPole-v0")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = PI(in_dim, out_dim)
    optimizer = th.optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(300):
        state = env.reset()[0]
        for t in range(200):
            action = pi.act(state)
            print(action, state, sep="-_|-_")
            state, reward, done, truncated, info = env.step(action)
            pi.rewards_list.append(reward)
            env.render()
            if done:
                break
        
        loss = train(pi, optimizer)
        print(pi.rewards_list)
        total_reward = sum(pi.rewards_list)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f"Epizode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}")

if __name__ == "__main__":
    main()
        
    
       