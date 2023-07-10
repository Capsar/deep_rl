import json

import gymnasium as gym
import numpy as np

from src.deep_rl.experiment import SACExperiment, MultiExperiment
from src.deep_rl.model import GRUSoftActorCritic, SoftActorCritic, TransformerSoftActorCritic, DLinearSoftActorCritic

SAC_CONFIG_PARAMS_LUNAR = {
    "max_steps": int(1E5),
    "replay_buffer_size": int(1E4),
    "batch_size": 128,
    "runner_steps": 10,
    "gamma": 0.995,
    "grad_repeats": 5,
    "alpha_lr": 0.001,
    "actor_lr": 0.001,
    "critic_lr": 0.002,
    "alpha": 1,
    "tau": 0.01,
    "render_every": 500,
    "print_every": 500,
    "save_path": "./data/experiments_data/sac_lunarlander/",
    "render": True,
    "hidden_size": [64, 64],
    "device": 'cpu'
}

SAC_CONFIG_PARAMS_PENDULUM = {
    "max_steps": 50000,
    "replay_buffer_size": 5000,
    "batch_size": 200,
    "runner_steps": 200,
    "gamma": 0.9,
    "grad_repeats": 32,
    "alpha_lr": 0.001,
    "actor_lr": 0.001,
    "critic_lr": 0.002,
    "alpha": 1,
    "tau": 0.01,
    "render_every": 1,
    "print_every": 1,
    "save_path": "./experiments_data/sac_pendulum/",
    "render": True,
    "hidden_size": [32, 32],
    "device": 'cpu'
}

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=np.nan)

    env = gym.make('LunarLanderContinuous-v2')

    obv_space = env.observation_space
    action_space = env.action_space
    SAC_Constructor = (obv_space, action_space, SAC_CONFIG_PARAMS_LUNAR['hidden_size'])

    print("Model constructor: ", SAC_Constructor)
    multi_experiment = MultiExperiment(4, env, SACExperiment, SoftActorCritic, SAC_Constructor, SAC_CONFIG_PARAMS_LUNAR)
    multi_experiment.run_experiments()
