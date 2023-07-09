"""
Controller:
 - train loop
 - target network
 - total loss

"""

import torch as th


class Controller:
    """
    Controller class that interacts with the environment using a PyTorch model.

    The Controller class provides an interface for a PyTorch model to interact with an environment.
    It takes as input a PyTorch model and a configuration parameters dictionary.
    It provides methods to choose actions based on the current state of the environment,
    both in a stochastic or deterministic manner.

    Attributes:
        model (th.nn.Module): A PyTorch model that decides which action to take based on the current state.
        config_params (dict): A dictionary of configuration parameters.
    """
    def __init__(self, model: th.nn.Module, config_params):
        """
        Initializes the Controller with the given PyTorch model and configuration parameters.

        Args:
            model (th.nn.Module): A PyTorch model that decides which action to take based on the current state.
            config_params (dict): A dictionary of configuration parameters.
        """
        self.model = model
        self.config_params = config_params

    def choose(self, x):
        """
        Chooses an action based on the current state using the PyTorch model.

        This method converts the input state to a PyTorch tensor, feeds it into the model, and returns the action
        decided by the model. The action is detached from the computation graph and converted to a NumPy array.

        Args:
            x (array_like): The current state of the environment.

        Returns:
            action (numpy.ndarray): The action chosen by the model based on the current state.
        """
        return self.model.act(th.tensor(x, dtype=th.float32)).detach().numpy()

    def get_deterministic_action(self, x):
        """
        Gets the deterministic action based on the current state using the PyTorch model.

        This method converts the input state to a PyTorch tensor, feeds it into the model's deterministic action
        function, and returns the action decided by the model. The action is detached from the computation graph and
        converted to a NumPy array.

        Args:
            x (array_like): The current state of the environment.

        Returns:
            action (numpy.ndarray): The deterministic action chosen by the model based on the current state.
        """
        return self.model.get_deterministic_action(th.tensor(x, dtype=th.float32)).detach().numpy()
