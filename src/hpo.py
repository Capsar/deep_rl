import itertools
import json
import warnings

import gym
import numpy as np
import pandas as pd

from src.deep_rl.experiment import SACExperiment, MultiExperiment
from src.deep_rl.sac_model import GRUSoftActorCritic, SoftActorCritic, TransformerSoftActorCritic, DLinearSoftActorCritic

def grid_search_params(params_dict):
    """
    Given a dictionary of hyperparameters, if a value is a list, loop over all values
    and create a grid search.
    """
    param_keys = params_dict.keys()
    param_values = params_dict.values()
    param_combinations = list(itertools.product(*[v if isinstance(v, list) else [v] for v in param_values]))
    for combination in param_combinations:
        yield dict(zip(param_keys, combination))

if __name__ == '__main__':
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    np.set_printoptions(suppress=True, linewidth=np.nan)
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")
    
    SAC_CONFIG_PARAMS_LUNAR = {
        "max_steps": int(1E5),
        "replay_buffer_size": int(1E4),
        "batch_size": [16, 32, 64],
        "runner_steps": [10, 25, 50],
        "gamma": 0.995,
        "grad_repeats": [3, 6, 9],
        "alpha_lr": 0.001,
        "actor_lr": 0.001,
        "critic_lr": 0.002,
        "alpha": 1,
        "tau": 0.01,
        "render_every": 1000,
        "print_every": 1000,
        "save_every": -1,
        "save_path": "./data/experiments_data/sac_lunarlander/",
        "render": True,
        "hidden_size": [[16, 16], [64, 64]]
    }

    gird_search_params_list = list(grid_search_params(SAC_CONFIG_PARAMS_LUNAR))

    for i, hyper_parms in enumerate(gird_search_params_list):
        print(f'{i}/{len(gird_search_params_list)}: {hyper_parms}')

        env = gym.make('LunarLanderContinuous-v2')

        obv_space = env.observation_space
        action_space = env.action_space
        SAC_Constructor = (obv_space, action_space, hyper_parms['hidden_size'])

        print("Model constructor: ", SAC_Constructor)
        multi_experiment = MultiExperiment(1, env, SACExperiment, SoftActorCritic, SAC_Constructor, hyper_parms)
        experiment_results_path = multi_experiment.run_experiments()