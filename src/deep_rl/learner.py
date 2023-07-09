"""
Learner

- train

"""
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn.functional as F

from src.deep_rl.model import SoftActorCritic
from src.deep_rl.replay_buffer import ReplayBuffer


def update_target_model(t_model, model, tau):
    for t_param, param in zip(t_model.parameters(), model.parameters()):
        t_param.data.copy_(tau * param.data + (1.0 - tau) * t_param.data)


class Learner:
    """
    Base Learner class for learning from a batch of experiences.

    The Learner class defines the interface for learning from a batch of experiences. It contains a model 
    and a dictionary of configuration parameters. It provides a method for training on a batch of experiences.

    Attributes:
        model (th.nn.Module): A PyTorch model that learns from the batch of experiences.
        config_params (dict): A dictionary of configuration parameters.
    """
    def __init__(self, model: th.nn.Module, config_params):
        """
        Initializes the Learner with the given PyTorch model and configuration parameters.

        Args:
            model (th.nn.Module): A PyTorch model that learns from the batch of experiences.
            config_params (dict): A dictionary of configuration parameters.
        """
        self.model = model
        self.config_params = config_params

    def train(self, batch: ReplayBuffer):
        """
        Trains the model on a batch of experiences.

        Args:
            batch (ReplayBuffer): A replay buffer containing a batch of experiences.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        return NotImplementedError


class SoftActorCriticLearner(Learner):
    """
    Soft Actor-Critic Learner class for learning from a batch of experiences.

    The SoftActorCriticLearner class is a specific type of Learner that uses a Soft Actor-Critic (SAC)
    model for learning from a batch of experiences. In addition to the base Learner's attributes, it also
    maintains a number of optimizers for different parts of the SAC model, an entropy target, and a device
    for running the computations.

    Attributes:
        model (SoftActorCritic): A Soft Actor-Critic model that learns from the batch of experiences.
        device (str): The device on which to run the computations ('cuda' if available, else 'cpu').
        tau (float): The rate at which to update the target network.
        entropy_target (float): The target for entropy maximization.
        log_alpha (th.Tensor): The log of alpha, used for entropy optimization.
        log_alpha_optimizer (th.optim.Adam): The optimizer for log_alpha.
        actor_optimizer (th.optim.Adam): The optimizer for the actor part of the model.
        critic1_optimizer (th.optim.Adam): The optimizer for the first critic part of the model.
        critic2_optimizer (th.optim.Adam): The optimizer for the second critic part of the model.
    """
    def __init__(self, model: SoftActorCritic, action_shape, config_params):
        """
        Initializes the SoftActorCriticLearner with the given SAC model, action shape, and configuration parameters.

        Args:
            model (SoftActorCritic): A Soft Actor-Critic model that learns from the batch of experiences.
            action_shape (tuple): The shape of the action space.
            config_params (dict): A dictionary of configuration parameters.
        """
        super().__init__(model, config_params)
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.tau = self.config_params['tau']

        # alpha, used for entropy optimization
        self.entropy_target = -np.prod(action_shape)
        self.log_alpha = th.tensor(np.log(self.config_params['alpha'])).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = th.optim.Adam([self.log_alpha], lr=self.config_params['alpha_lr'])
        self.actor_optimizer = th.optim.Adam(self.model.actor.parameters(), lr=self.config_params['actor_lr'])
        self.critic1_optimizer = th.optim.Adam(self.model.critic1.parameters(), lr=self.config_params['critic_lr'])
        self.critic2_optimizer = th.optim.Adam(self.model.critic2.parameters(), lr=self.config_params['critic_lr'])

    def calc_q_target(self, batch: ReplayBuffer):
        """
        Calculates the Q-value targets for the given batch of experiences.

        Args:
            batch (ReplayBuffer): A replay buffer containing a batch of experiences.

        Returns:
            targets (th.Tensor): The Q-value targets for the given batch of experiences.
        """
        with th.no_grad():
            next_action, next_log_pi = self.model.sample_log_prob(batch['next_states'])
            entropy = self.log_alpha.exp() * next_log_pi.sum(dim=-1, keepdim=True)
            t_q_values = th.min(*self.model.q_values(batch['next_states'], next_action, target=True))
            targets = batch['rewards'] + self.config_params['gamma'] * (1. - batch['dones']) * (t_q_values - entropy)
        return targets

    def train(self, batch: ReplayBuffer):
        """
        Trains the model on a batch of experiences.

        This method calculates the policy loss, the two Q-value losses, and the alpha loss. It updates the model 
        parameters using the respective optimizers and also updates the target model parameters.

        Args:
            batch (ReplayBuffer): A replay buffer containing a batch of experiences.

        Returns:
            policy_loss (float): The average policy loss over the batch.
            q1_loss (float): The average first Q-value loss over the batch.
            q2_loss (float): The average second Q-value loss over the batch.
            alpha_loss (float): The alpha loss.
        """
        action, log_pi = self.model.sample_log_prob(batch['states'])
        entropy = self.log_alpha.exp() * log_pi
        q_values = th.min(*self.model.q_values(batch['states'], action))
        policy_loss = (entropy - q_values).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        q1_values, q2_values = self.model.q_values(batch['states'], batch['actions'])
        td_targets = self.calc_q_target(batch)
        q1_loss = F.smooth_l1_loss(q1_values, td_targets).mean()
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()

        q2_loss = F.smooth_l1_loss(q2_values, td_targets).mean()
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (log_pi + self.entropy_target).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        update_target_model(self.model.t_critic1, self.model.critic1, self.tau)
        update_target_model(self.model.t_critic2, self.model.critic2, self.tau)
        return policy_loss.mean().item(), q1_loss.mean().item(), q2_loss.mean().item(), alpha_loss.mean().item()
