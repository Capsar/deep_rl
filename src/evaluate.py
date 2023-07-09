import json

import gym
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from src.deep_rl.controller import Controller
import os

def evaulate_single_experiment_model(env, experiment_folder, experiment_model, results_name, render_all=False):
    results = json.load(open(f"{experiment_folder}{results_name}"))
    print('Duration: ', duration:=np.max(results['n_episode_durations']))
    config_params = json.load(open(f"{experiment_folder}config.json"))
    model = th.load(f"{experiment_folder}{experiment_model}")
    model.eval()
    controller = Controller(model, config_params)
    s, _ = env.reset()
    for step in range(1000):
        a = controller.get_deterministic_action(s)
        ns, r, t, d, _ = env.step(a)
        s = ns
        env.render()
        if d or t:
            break

def list_files_in_subfolders(folder_path):
    folders_and_files = {}
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            print(f"Files in {dir_path}:")
            folders_and_files[dir_path] = []
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    print(f"\t{file}")
                    folders_and_files[dir_path].append(file)
    return folders_and_files

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=np.nan)

    folders_and_files = list_files_in_subfolders('./data/experiments_data/sac_lunarlander/')
    env = gym.make('LunarLanderContinuous-v2', render_mode='human')

    for key, files in folders_and_files.items():
        model_file = None
        results_file = None
        for file in files:
            if file.endswith('.pth'):
                model_file = file
            elif file.startswith('results') and file.endswith('.json'):
                results_file = file
        
        if model_file is None or results_file is None:
            continue

        evaulate_single_experiment_model(env, key+'/', model_file, results_file)