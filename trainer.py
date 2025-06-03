import torch
import torch.optim as optim
import time
import numpy as np
import random
import os
import math
from PPO import PPO
from datetime import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from args import *
from cartpole_env import InvertedPendulumEnv


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, experiment_num)
tensorboard_path = directory + "PPO_{}_{}.pth".format(env_name, experiment_num) 


def trainer_ppo(env):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ppo_agent = PPO(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, g_lambda, T_horizon, T_episode, batch_size ,device)
    writer = SummaryWriter(log_dir=tensorboard_path)
    start_time = datetime.now().replace(microsecond=0)

    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    i_writer = 0
    i_writer_reward = 0

    while time_step <= max_training_timesteps:
        
        state = env.reset_model()
        
        current_ep_reward = 0

        # playing one episode
        while True:
            
            action = ppo_agent.choose_action(state)
            
            state, reward, done = env.step(action)

            ppo_agent.buffer.rewards.append(reward)

            ppo_agent.buffer.is_terminals.append(done)
            
            time_step +=1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                # updating the policy 
                policy_loss, value_loss, entropy, returns , gae = ppo_agent.update()
                print("f")
                writer.add_scalar('policy_loss', policy_loss, i_writer)
                writer.add_scalar('value_loss', value_loss, i_writer)
                writer.add_scalar('entropy', entropy, i_writer)
                writer.add_scalar('returns', returns, i_writer)
                writer.add_scalar('advantages', gae, i_writer)
                
                i_writer+=1

            if time_step % print_freq == 0:
                # log current reward 
                print_avg_reward = print_running_reward / print_running_episodes
                writer.add_scalar('reward', print_avg_reward, i_writer_reward)
                print_running_reward = 0
                i_writer_reward +=1
                print_running_episodes = 0
                
            if time_step % save_model_freq == 0:
                #save checkpoint
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + "PPO_{}_{}_{}_{}.pth".format(env_name, experiment_num, i_episode, print_avg_reward)
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1


    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at : ", start_time)
    print("Finished training at : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    env = InvertedPendulumEnv(reward_mode = reward_mode, T_max = T_episode )
    trainer_ppo(env)