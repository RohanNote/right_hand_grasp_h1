#!/usr/bin/env python3
"""
Vision-based Grasping Policy Training
Progressive reward structure with proper episode length and sensor integration
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime

# Import our custom components
from vision_grasping_env import VisionGraspingEnv
from enhanced_cnn_policy import EnhancedCNNPolicy


class PPOTrainer:
    """
    PPO Trainer for vision-based grasping with progressive rewards
    """
    
    def __init__(self, 
                 env: VisionGraspingEnv,
                 policy: EnhancedCNNPolicy,
                 lr: float = 3e-4,
                 cnn_lr: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        
        # PPO hyperparameters
        self.lr = lr
        self.cnn_lr = cnn_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizers (different learning rates for CNN vs MLP)
        self.optimizer = optim.Adam([
            {'params': self.policy.vision_cnn.parameters(), 'lr': cnn_lr},
            {'params': self.policy.spatial_attention.parameters(), 'lr': cnn_lr},
            {'params': self.policy.feature_extractor.parameters(), 'lr': cnn_lr},
            {'params': self.policy.proprio_mlp.parameters(), 'lr': lr},
            {'params': self.policy.combined_mlp.parameters(), 'lr': lr},
            {'params': self.policy.action_mean.parameters(), 'lr': lr},
            {'params': self.policy.action_log_std.parameters(), 'lr': lr},
        ])
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.training_log = []
        
        print(f" PPO Trainer initialized")
        print(f"    Device: {device}")
        print(f"    Learning rates: CNN={cnn_lr}, MLP={lr}")
        print(f"    Environment: {env.__class__.__name__}")
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def collect_rollout(self, num_steps: int):
        """Collect experience using current policy"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        obs, _ = self.env.reset()  # Unpack tuple (observation, info)
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get action from policy
            with torch.no_grad():
                action_mean, action_log_std = self.policy(obs_tensor)
                action_dist = Normal(action_mean, torch.exp(action_log_std))
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                
                # Simple value estimation (using action mean as value proxy)
                value = torch.norm(action_mean, dim=-1)
            
            # Step environment
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.flatten()  # Ensure 1D array
            next_obs, reward, done, truncated, info = self.env.step(action_np)
            
            # Store experience
            observations.append(obs)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Episode finished
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(info.get('success', False))
                
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, "
                      f"Success={info.get('success', False)}")
                
                # Reset for next episode
                obs, _ = self.env.reset()  # Unpack tuple (observation, info)
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        # Convert to tensors
        observations = self._obs_to_tensor_batch(observations)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        return observations, actions, rewards, dones, values, log_probs
    
    def _obs_to_tensor(self, obs):
        """Convert single observation to tensor"""
        return {
            'image': torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0,
            'proprioception': torch.FloatTensor(obs['proprioception']).unsqueeze(0).to(self.device),
            'contact': torch.FloatTensor(obs['contact']).unsqueeze(0).to(self.device),
            'object_pos': torch.FloatTensor(obs['object_pos']).unsqueeze(0).to(self.device),
        }
    
    def _obs_to_tensor_batch(self, obs_list):
        """Convert batch of observations to tensor"""
        batch_size = len(obs_list)
        
        # Convert to numpy arrays first to avoid tensor creation warning
        images = np.array([obs['image'] for obs in obs_list])
        proprioceptions = np.array([obs['proprioception'] for obs in obs_list])
        contacts = np.array([obs['contact'] for obs in obs_list])
        object_positions = np.array([obs['object_pos'] for obs in obs_list])
        
        return {
            'image': torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device) / 255.0,
            'proprioception': torch.FloatTensor(proprioceptions).to(self.device),
            'contact': torch.FloatTensor(contacts).to(self.device),
            'object_pos': torch.FloatTensor(object_positions).to(self.device),
        }
    
    def update_policy(self, observations, actions, rewards, dones, values, log_probs, epochs=4):
        """Update policy using PPO"""
        # Clear GPU cache before processing
        torch.cuda.empty_cache()
        
        # Compute advantages and returns
        with torch.no_grad():
            next_value = torch.norm(self.policy(observations)[0], dim=-1).mean()
        
        advantages, returns = self.compute_gae(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy(),
            next_value.cpu().numpy()
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        for epoch in range(epochs):
            # Forward pass
            action_mean, action_log_std = self.policy(observations)
            action_dist = Normal(action_mean, torch.exp(action_log_std))
            
            new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)
            
            # Simple value estimation
            new_values = torch.norm(action_mean, dim=-1)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(new_values, returns)
            entropy_loss = -entropy.mean()
            
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear GPU cache after updates
        torch.cuda.empty_cache()
    
    def train(self, total_timesteps: int = 1000000, 
              rollout_steps: int = 2048,
              save_interval: int = 10000,
              log_interval: int = 1000):
        """Main training loop"""
        
        print(f" Starting training for {total_timesteps} timesteps")
        print(f"    Rollout steps: {rollout_steps}")
        print(f"    Save interval: {save_interval}")
        print(f"    Log interval: {log_interval}")
        
        start_time = time.time()
        timestep = 0
        
        while timestep < total_timesteps:
            # Collect rollout
            observations, actions, rewards, dones, values, log_probs = self.collect_rollout(rollout_steps)
            
            # Update policy
            self.update_policy(observations, actions, rewards, dones, values, log_probs)
            
            timestep += rollout_steps
            
            # Logging
            if timestep % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                success_rate = np.mean(self.success_rates) if self.success_rates else 0
                
                elapsed_time = time.time() - start_time
                fps = timestep / elapsed_time
                
                print(f"\n Training Progress - Timestep {timestep}")
                print(f"    Avg Reward: {avg_reward:.2f}")
                print(f"    Avg Length: {avg_length:.1f}")
                print(f"    Success Rate: {success_rate:.2%}")
                print(f"    FPS: {fps:.1f}")
                print(f"    Elapsed: {elapsed_time/3600:.1f}h")
                
                # Save training log
                self.training_log.append({
                    'timestep': timestep,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'success_rate': success_rate,
                    'elapsed_time': elapsed_time
                })
            
            # Save model
            if timestep % save_interval == 0:
                self.save_model(f"policy_checkpoint_{timestep}.pth")
                self.save_training_log(f"training_log_{timestep}.json")
        
        print(f"\n Training completed!")
        print(f"    Total time: {(time.time() - start_time)/3600:.1f}h")
        print(f"    Final avg reward: {np.mean(self.episode_rewards):.2f}")
        print(f"    Final success rate: {np.mean(self.success_rates):.2%}")
        
        # Save final model
        self.save_model("final_policy.pth")
        self.save_training_log("final_training_log.json")
    
    def save_model(self, filename: str):
        """Save policy model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log,
        }, filename)
        print(f" Model saved: {filename}")
    
    def save_training_log(self, filename: str):
        """Save training log"""
        with open(filename, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f" Training log saved: {filename}")


def main():
    """Main training function"""
    print(" Starting Vision-based Grasping Policy Training")
    
    # Create environment
    env = VisionGraspingEnv(
        model_path="scene_h1_allegro_new.xml",
        camera_name="head_camera",
        camera_width=640,
        camera_height=480,
        max_episode_steps=500,  # Longer episodes for proper learning
        render_mode=None  # No rendering during training for speed
    )
    
    # Create policy
    policy = EnhancedCNNPolicy(
        camera_width=640,
        camera_height=480,
        n_joints=21,  # 5 arm + 16 hand joints
        n_contact_sensors=5,
        n_object_dims=3,
        action_dim=21,  # 5 arm + 16 hand joints
        hidden_dim=256
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=3e-4,
        cnn_lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    )
    
    # Start training
    trainer.train(
        total_timesteps=1000000,  # 1M timesteps
        rollout_steps=4096,       # Optimized for A100 GPUs (80GB memory)
        save_interval=5000,       # Save every 5K steps (~10 episodes)
        log_interval=500          # Log every 500 steps (~1 episode)
    )
    
    env.close()


if __name__ == "__main__":
    main()
