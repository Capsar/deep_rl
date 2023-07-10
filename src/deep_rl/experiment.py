"""
Experiment class for running experiments

- rollouts
- graph drawing

"""
import json
import os
from datetime import datetime

import numpy as np
import torch as th
from cycler import cycler
from gymnasium import Env

from src.deep_rl import timed_decorator
from src.deep_rl.learner import Learner, SoftActorCriticLearner
from src.deep_rl.replay_buffer import ReplayBuffer
from src.deep_rl.runner import Runner
from src.deep_rl.controller import Controller
import matplotlib.pyplot as plt


class MultiExperiment:

    def __init__(self, n, env, experiment_class, model_class, model_constructor, config_params):
        self.config_params = config_params
        number_of_graph_points = int(self.config_params['max_steps'] / self.config_params['runner_steps'])
        self.n = n

        self.experiments_buffer_shapes = {'n_rewards': (number_of_graph_points,),
                                          'n_episode_lengths': (number_of_graph_points,),
                                          'n_episode_infos': (number_of_graph_points,),
                                          'n_loss_means': (number_of_graph_points, 4),
                                          'n_loss_stds': (number_of_graph_points, 4),
                                          'n_env_steps': (number_of_graph_points,),
                                          'n_episode_durations': (number_of_graph_points,)}
        self.experiments_buffer = ReplayBuffer(n, self.experiments_buffer_shapes)
        self.env = [env for _ in range(n)]
        self.models = [model_class(*model_constructor) for _ in range(n)]
        self.experiment_class = experiment_class
        self.experiment_folder_path = self.config_params['save_path'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.experiment_folder_path)
        json.dump(self.config_params, open(f'{self.experiment_folder_path}/config.json', 'w'))

    def run_experiments(self):
        for i in range(self.n):
            experiment = self.experiment_class(self.env[i], model=self.models[i], config_params=self.config_params)
            try:
                fig, rewards, episode_lengths, episode_infos, episode_loss_means, episode_loss_stds, env_steps, episode_durations = experiment.run()
                self.experiments_buffer.store(n_rewards=rewards, n_episode_lengths=episode_lengths, n_episode_infos=episode_infos,
                                              n_loss_means=episode_loss_means, n_loss_stds=episode_loss_stds,
                                              n_env_steps=env_steps,
                                              n_episode_durations=np.array([sum(episode_durations[:i]) for i in range(len(episode_durations))]))
                th.save(experiment.learner.model, f'{self.experiment_folder_path}/experiment_{i}.pth')
                self.experiments_buffer.save_to_file(f'{self.experiment_folder_path}/results_{i}_{np.round(np.max(rewards[-2000:]), 2)}_{np.round(np.max(episode_infos[-2000:]), 2)}.json')
                fig.savefig(f'{self.experiment_folder_path}/experiment_{i}.png')
                plt.close(fig)
            except KeyboardInterrupt:
                experiment.close()
        plt.ioff()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        n_env_steps = self.experiments_buffer['n_env_steps']
        n_rewards = self.experiments_buffer['n_rewards']
        n_loss_means = self.experiments_buffer['n_loss_means']
        n_episode_durations = self.experiments_buffer['n_episode_durations']
        n_episode_infos = self.experiments_buffer['n_episode_infos']

        for i in range(self.n):
            axes[0].plot(n_env_steps[i], n_rewards[i], c='b', alpha=0.1)
        axes[0].fill_between(np.mean(n_env_steps, axis=0), np.mean(n_rewards, axis=0) - np.std(n_rewards, axis=0),
                             np.mean(n_rewards, axis=0) + np.std(n_rewards, axis=0), alpha=0.4)
        axes[0].plot(np.mean(n_env_steps, axis=0), np.mean(n_rewards, axis=0), c='r')
        axes[0].set_title('Mean Total Rewards')

        axes[1].set_prop_cycle(cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']))
        for i in range(self.n):
            axes[1].plot(n_env_steps[i], n_loss_means[i], alpha=0.1)
        axes[1].plot(np.mean(n_env_steps, axis=0), np.mean(n_loss_means, axis=0))
        axes[1].set_title('Mean Losses')

        for i in range(self.n):
            axes[2].plot(n_env_steps[i], n_episode_infos[i], c='b', alpha=0.1)
        axes[2].plot(np.mean(n_env_steps, axis=0), np.mean(n_episode_infos, axis=0), c='tab:blue')
        axes[2].set_title('Mean Episode Infos')
        plt.savefig(f'{self.experiment_folder_path}/all_results.png')
        # plt.show()
        plt.close(fig)
        return self.experiment_folder_path


class Experiment:

    def __init__(self, config_params, render_graphs=True):

        self.learner = None
        self.controller = None
        self.runner = None

        self.config_params = config_params
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = {}
        self.episode_loss_means = []
        self.episode_loss_stds = []
        self.env_steps = []
        self.episode_runner_durations = []
        self.episode_learn_durations = []
        self.episode_durations = []
        self.temp_time = datetime.now()
        self.start_time = datetime.now()

        self.render_graphs = render_graphs
        if self.render_graphs:
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

            self.axs[0].plot(self.env_steps, self.episode_rewards)
            self.axs[0].set_prop_cycle(cycler('color', ['tab:blue']))
            self.axs[0].set_ylabel('Episode Sum of Rewards')
            self.info_ax, self.info_labels = None, None

            self.axs[1].plot(self.env_steps, self.episode_lengths)
            self.axs[1].set_prop_cycle(cycler('color', ['tab:blue']))
            self.axs[1].set_ylabel('Episode Lengths')

            labels = ['Policy', 'Critic_1', 'Critic_2', 'Alpha']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

            # Create and plot the lines using a list comprehension and zip the labels and colors
            for label, color in zip(labels, colors):
                self.axs[2].plot([0], [0], label=label, color=color)
            self.axs[2].set_ylabel('Episode Losses')
            self.axs[2].legend()

            for ax in self.axs:
                ax.set_xlabel('Environment Steps')
            self.first_run = True
            plt.tight_layout()
            plt.pause(1)

    def plot(self, step):
        if not self.render_graphs:
            return

        if self.first_run and len(self.episode_infos) > 0:
            self.info_ax = self.axs[0].twinx()
            for key in self.episode_infos.keys():
                self.info_ax.plot(self.env_steps, self.episode_infos[key], label=key, alpha=0.5, color='tab:orange')
            self.info_ax.legend()
            plt.tight_layout()

        if self.config_params['render'] and len(self.episode_rewards) > 0:
            if step % self.config_params['render_every'] == 0:
                self.axs[0].lines[0].set_xdata(self.env_steps)
                self.axs[0].lines[0].set_ydata(self.episode_rewards)
                self.axs[0].draw_artist(self.axs[0].lines[0])

                self.axs[1].lines[0].set_xdata(self.env_steps)
                self.axs[1].lines[0].set_ydata(self.episode_lengths)
                self.axs[1].draw_artist(self.axs[1].lines[0])

                for i in range(len(self.axs[2].lines)):
                    self.axs[2].lines[i].set_xdata(self.env_steps)
                    self.axs[2].lines[i].set_ydata(np.asarray(self.episode_loss_means)[:, i])
                    self.axs[2].draw_artist(self.axs[2].lines[i])

                if self.info_ax is not None:
                    for line, key in zip(self.info_ax.lines, self.episode_infos.keys()):
                        line.set_xdata(self.env_steps)
                        line.set_ydata(self.episode_infos[key])
                    for line in self.info_ax.lines:
                        self.info_ax.draw_artist(line)
                    self.info_ax.relim(visible_only=True)
                    self.info_ax.autoscale_view(scalex=True, scaley=True)

                # Update the axes limits
                for ax in self.axs:
                    ax.relim(visible_only=True)
                    ax.autoscale_view(scalex=True, scaley=True)

                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()

    def record(self, returns, rollout_lengths, rollout_infos, episode_loss_mean, episode_loss_std, step, run_duration, learn_duration):
        if len(rollout_infos) > 0:
            for key in rollout_infos.shape.keys():
                key_values_mean = np.asarray(rollout_infos[key]).mean()
                if key not in self.episode_infos.keys():
                    self.episode_infos[key] = []
                self.episode_infos[key].append(key_values_mean)

        self.episode_loss_means.append(episode_loss_mean)
        self.episode_loss_stds.append(episode_loss_std)
        self.env_steps.append(step)
        self.episode_runner_durations.append(run_duration)
        self.episode_learn_durations.append(learn_duration)
        self.episode_durations.append(run_duration + learn_duration)
        return_string = ""
        length_string = ""

        # Calculate the mean of episode rewards and lengths if provided
        mean_rewards = np.asarray(returns).sum(axis=-1).mean() if returns is not None else None
        mean_lengths = np.asarray(rollout_lengths).mean() if rollout_lengths is not None else None
        # If there are no existing episode rewards, initialize the lists with the given mean values
        if len(self.episode_rewards) == 0 and mean_rewards is not None and mean_lengths is not None:
            num_entries_to_add = len(self.episode_loss_means) - 1
            self.episode_rewards.extend([mean_rewards] * num_entries_to_add)
            self.episode_lengths.extend([mean_lengths] * num_entries_to_add)
        # If there are existing episode rewards, update the lists with the new mean values
        if len(self.episode_rewards) > 0:
            if mean_rewards is None and mean_lengths is None:
                self.episode_rewards.append(self.episode_rewards[-1])
                self.episode_lengths.append(self.episode_lengths[-1])
            else:
                self.episode_rewards.append(mean_rewards)
                self.episode_lengths.append(mean_lengths)
            return_string = f"Return: {self.episode_rewards[-1]}"
            length_string = f"Length: {self.episode_lengths[-1]}"

        step_string = f"Steps: {step} / {len(self.episode_rewards)}"
        rn = self.config_params['print_every'] // self.config_params['runner_steps']
        duration_string = f"({np.round(np.sum(self.episode_runner_durations[-rn:]), 4)} / {np.round(np.sum(self.episode_learn_durations[-rn:]), 4)})"
        loss_string = f"Loss: {self.episode_loss_means[-1]} +/- {self.episode_loss_stds[-1]})"
        if step % self.config_params['print_every'] == 0:
            print(f"{step_string} {duration_string} {return_string} {length_string} {loss_string}")

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    def run(self):
        if len(self.episode_infos) == 0:
            self.episode_infos = {'info': [np.linspace(0, len(self.episode_rewards), len(self.episode_rewards))]}
        return self.fig, self.episode_rewards, self.episode_lengths, self.episode_infos[list(self.episode_infos.keys())[0]], self.episode_loss_means, self.episode_loss_stds, self.env_steps, self.episode_durations


class SACExperiment(Experiment):

    def __init__(self, env: Env, model: th.nn.Module, config_params):
        super().__init__(config_params)
        self.env = env
        self.learner = SoftActorCriticLearner(model, self.env.action_space.shape, config_params)
        self.controller = Controller(model, config_params)
        self.runner = Runner(env, self.controller, config_params)
        self.replay_buffer_shapes = {'states': self.env.observation_space.shape,
                                     'actions': self.env.action_space.shape,
                                     'rewards': (1,),
                                     'next_states': self.env.observation_space.shape,
                                     'dones': (1,)}
        self.replay_buffer = ReplayBuffer(config_params['replay_buffer_size'], self.replay_buffer_shapes, config_params['batch_size'])
        self.grad_repeats = config_params['grad_repeats']

    @timed_decorator
    def _learn_from_episodes(self, episodes_buffer):
        self.replay_buffer.append(episodes_buffer)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]

        loss = []
        for _ in range(self.grad_repeats):
            batch_buffer = self.replay_buffer.sample()
            policy_loss, q1_loss, q2_loss, alpha_loss = self.learner.train(batch_buffer)
            loss.append([policy_loss, q1_loss, q2_loss, alpha_loss])
        return np.mean(loss, axis=0), np.std(loss, axis=0)

    def run(self):
        step = 0
        while step < self.config_params['max_steps']:
            episode_buffer, rollout_returns, rollout_lengths, rollout_infos, runner_duration = self.runner.run_steps(self.config_params['runner_steps'])
            episode_loss_mean, episode_loss_std, learn_duration = self._learn_from_episodes(episode_buffer)
            step += self.config_params['runner_steps']
            self.record(rollout_returns, rollout_lengths, rollout_infos, episode_loss_mean, episode_loss_std, step, runner_duration, learn_duration)
            self.plot(step)
            self.first_run = False
        return super().run()
