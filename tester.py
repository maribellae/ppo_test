import torch
import torch.optim as optim
import time
import numpy as np
import random
import os
from PPO import PPO
from args import *
from cartpole_env import InvertedPendulumEnv

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


tensorboard_path = directory + "PPO_{}_{}.pth".format(env_name, experiment_num) 


def tester_ppo(env):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ppo_agent = PPO(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, g_lambda, T_horizon, T_episode, batch_size ,device)
    ppo_agent.load(checkpoint_path)
    
    for _ in range (10):

        state = env.reset_model()
        
        while True:
            
            action = ppo_agent.choose_action(state)
            
            state, reward, done = env.step(action)
            time.sleep(0.01)

            if done:
                break


if __name__ == '__main__':
    env = InvertedPendulumEnv(reward_mode = reward_mode, T_max = T_episode )
    tester_ppo(env)