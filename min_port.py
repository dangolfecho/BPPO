import os
import sys
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import PyFlyt.gym_envs

import minari
from minari import DataCollector
import time

from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization

def collate_fn(batch):
    return {
        "id":  torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x.observations) for
                x in batch], batch_first=True),
        "actions": torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x.actions) for
                x in batch], batch_first=True),
        "rewards": torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x.rewards) for
                x in batch], batch_first=True),
        "terminations": torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x.terminations) for
                x in batch], batch_first=True),
        "truncations": torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x.truncations) for
                x in batch], batch_first=True),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Experiment
    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(2e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(2e6), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e6), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) # try to reduce the bc/q/v step if it works poorly, 5e-4/2e-5/2e-5 for bc/q/v, for example
    parser.add_argument("--bc_hidden_dim", default=1024, type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(1e3), type=int)
    parser.add_argument("--bppo_hidden_dim", default=1024, type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0.0, type=float) # for ()-medium-() tasks, try to use the entropy loss, weight == 0.01
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=True, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=False, type=bool)
    
    args = parser.parse_args()
    print(f'------current env {args.env} and current seed {args.seed}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    env = gym.make(args.env)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    dataset_id = ((args.env).split('/'))[1] + '/' + 'dataset-14-0-20-v3'
    dataset = minari.load_dataset(dataset_id)
    max_size = len(dataset) + dataset.total_steps
    replay_buffer = OfflineReplayBuffer(device=device,
            state_dim=observation_dim,
            action_dim=action_dim,
            max_size=max_size)

    replay_buffer.load_dataset(dataset=dataset)
    exit()
