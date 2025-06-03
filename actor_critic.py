import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_dim = action_dim

        # predicting the mean
        self.actor = nn.Sequential(
                            self.layer_init(nn.Linear(state_dim, 64)),
                            nn.Tanh(),
                            self.layer_init(nn.Linear(64, 64)),
                            nn.Tanh(),
                            self.layer_init(nn.Linear(64, action_dim), std=0.01),     
                        )
        
        self.critic = nn.Sequential(
                        self.layer_init(nn.Linear(state_dim, 64)),
                        nn.Tanh(),
                        self.layer_init(nn.Linear(64, 64)),
                        nn.Tanh(),
                        self.layer_init(nn.Linear(64, 1), std=1.0)
                    )
        # learnable variance 
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    # for PPO training 
    def forward(self, state, action):

        action_mean = self.actor(state)

        action_var = self.log_std.exp().expand_as(action_mean)

        dist = torch.distributions.Normal(action_mean, action_var)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action).sum(1)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
        
    # for acting in a loop
    def act_env(self, state):

        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    @staticmethod
    def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        nn.init.orthogonal_(layer.weight, gain = std)
        layer.bias.data.fill_(0.0)
        return layer