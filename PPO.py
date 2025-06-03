import torch 
import torch.nn as nn
import torch.nn.functional as F
from actor_critic import ActorCritic
from ppo_rollout_buffer import PPO_RolloutBuffer
from torch.optim.lr_scheduler import  ExponentialLR
import numpy as np
from copy import deepcopy
from welford import Welford
from torch.utils.data import ConcatDataset

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, g_lambda, horizon, max_T, batch_size, device):
        self.device = device
        self.gamma = gamma
        self.horizon = horizon
        self.max_T = max_T
        self.eps_clip = eps_clip
        self.g_lambda = g_lambda
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.obs_stats = Welford() # for observation normalization
        
        self.buffer = PPO_RolloutBuffer() # trajectories buffer

        self.policy = ActorCritic(state_dim, action_dim, self.device).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},  
        ])

        self.policy_old = ActorCritic(state_dim, action_dim,self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.scheduler = ExponentialLR(self.optimizer, gamma = 0.99 )
        
    def normalize_observation(self, obs, update=False):
        
        if update:
            self.obs_stats.add(deepcopy(obs))

        if np.isnan(self.obs_stats.var_s).any() or (0 in self.obs_stats.var_s):
            return np.clip((obs-self.obs_stats.mean),-10,10)
        else:
            return np.clip((obs-self.obs_stats.mean)/np.sqrt(self.obs_stats.var_s),-10,10)

    def choose_action(self, state):
        # acting in the environment + adding trajectories to the buffer 

        state = self.normalize_observation(state, True)  # normalization of observations
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act_env(state)
                
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()


    def update(self):
        
        gae = []
        returns = []
        discounted = 0
        
        # calculating of the GAE (generalized advantage estimate) and Returns
        last_value = self.buffer.state_values[-1]
        for reward, is_terminal, value in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),reversed(self.buffer.state_values)):
            
            if is_terminal:
                discounted = 0
                discounted_v = 0
                last_value = 0

            discounted = reward + (1-is_terminal)*(self.gamma* self.g_lambda * discounted)  - value + self.gamma*last_value
            discounted_v = reward + (1-is_terminal)*(self.gamma* self.g_lambda * discounted) + self.gamma*last_value
            gae.append(discounted)
            returns.append( discounted_v)
            last_value = value

        returns.reverse()
        gae.reverse()
 
        gae = torch.tensor(gae, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # creating the dataset from the collected trajectories
        datasets_list = []
        idxs = np.random.permutation(self.max_T * self.horizon)

        for start in range(0, self.max_T * self.horizon, self.batch_size):
            mb = idxs[start : start + self.batch_size]
            temp_dataset = torch.utils.data.TensorDataset (
                old_states[mb],
                old_actions[mb],
                old_logprobs[mb],
                gae[mb],
                old_state_values[mb],
                returns[mb]
            )

            datasets_list.append(temp_dataset)

        dataloader = torch.utils.data.DataLoader(ConcatDataset(datasets_list), batch_size=self.batch_size)


        # training for K epoches
        policy_losses, value_losses, entropies = [], [], []
        
        for _ in range(self.K_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_logprobs, batch_gae, state_values_batch, batch_returns = batch
                
                batch_gae =(batch_gae - batch_gae.mean()) /  torch.clamp(batch_gae.std(),min=1e-5)
                
                logprobs, state_values, dist_entropy = self.policy(batch_states, batch_actions)

                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs - batch_old_logprobs.detach())

                surr1 = ratios * batch_gae

                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_gae
                
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (state_values -batch_returns).pow(2).mean()
                entropy_loss = dist_entropy.mean()
                
                loss = policy_loss +  value_loss  #- 0.005 *  entropy_loss 

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())
                value_losses.append(value_loss.item())
        
        
        print(f"Policy loss: { sum(policy_losses) / len(policy_losses)}, Value loss: {sum(value_losses) / len(value_losses)}, Entropy loss: {sum(entropies) / len(entropies)}")
        
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > 3e-6:
            self.scheduler.step()
 
        print('Learning rate: ', self.optimizer.param_groups[0]["lr"])
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return (sum(policy_losses) / len(policy_losses), sum(value_losses) / len(value_losses), sum(entropies) / len(entropies), returns.mean() , gae.mean())
    
    
    def save(self, checkpoint_path):
        torch.save({'model' :self.policy_old.state_dict(), 'obs_stats' : self.obs_stats}, checkpoint_path)
   

    def load(self, checkpoint_path):
        torch.serialization.add_safe_globals([Welford])
        checkpoint = torch.load(checkpoint_path,weights_only=False,map_location=self.device)
        self.policy_old.load_state_dict(checkpoint['model'])
        self.policy.load_state_dict(checkpoint['model'])
        self.obs_stats = checkpoint['obs_stats']
