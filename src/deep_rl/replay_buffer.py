import json

import torch as th

class ReplayBuffer:
    """
    A class that implements a replay buffer for reinforcement learning. The replay buffer is used to store
    experiences, i.e., states, actions, rewards, next states, etc., that the agent can learn from. It is also
    used in storing the final results in the experiments.

    Attributes:
        device (str): Device to use for computations ('cuda' if available else 'cpu').
        buffer_size (int): Maximum size of the replay buffer.
        shapes (dict): Dictionary mapping keys to shape of the corresponding experiences.
        batch_size (int, optional): Batch size to use for sampling. Default is 0.
        position (int): Current position in the replay buffer.
        dict (dict): Dictionary storing the experiences.
    """
    def __init__(self, buffer_size, shapes, batch_size=0, device='cpu'):
        """
        Constructor for the ReplayBuffer class.
        
        Args:
            buffer_size (int): The maximum size of the buffer.
            shapes (dict): The shapes of the data to be stored in the buffer. Keys are strings denoting data type (e.g., 'states', 'actions') and values are tuples representing data dimensions.
            batch_size (int, optional): The number of samples to return when 'sample' method is called. If not provided, no data will be returned by the 'sample' method.
        """
        self.device = device
        self.buffer_size = buffer_size
        self.shapes = shapes
        self.batch_size = batch_size
        self.position = 0
        self.dict = {}
        for key, shape in shapes.items():
            self.dict[key] = th.zeros((buffer_size, *shape), device=self.device)
        self.shape = {key: value.shape for key, value in self.dict.items()}

    def __str__(self):
        """Returns a string representation of the buffer dictionary."""
        return self.dict.__str__()

    def __len__(self):
        """Returns the current size of the buffer."""
        return min(self.buffer_size, self.position)

    def __getitem__(self, key):
        """Allows indexing into the buffer to get specific types of data."""
        return self.dict[key]

    def value_to_list(self, key):
        """
        Returns the stored values for a specific key as a list.

        Args:
            key (str): The key for the stored values.

        Returns:
            list: The stored values as a list.
        """
        return self.dict[key].cpu().numpy().tolist()

    def pos(self):
        """
        Returns the current position in the buffer.

        Returns:
            int: The current position in the buffer.
        """
        return self.position % self.buffer_size

    def add_value(self, key, value):
        """
        Adds value(s) to the buffer at the current position.

        Args:
            key (str): The key to add the value(s) to.
            value (Tensor): The value(s) to add.
        """
        if self.pos() + len(value) > self.buffer_size:
            self.dict[key][self.pos():] = value[:self.buffer_size - self.pos()]
            self.dict[key][:len(value) - (self.buffer_size - self.pos())] = value[self.buffer_size - self.pos():]
        else:
            self.dict[key][self.pos():self.pos() + len(value)] = value

    def store(self, **kwargs):
        """
        Stores a single experience in the buffer. The keys in the dictionary should match those defined when initializing the buffer.

        Args:
            **kwargs: The keys are strings representing the types of data (e.g., 'states', 'actions') and the values are the data to store.
        """
        for key, value in kwargs.items():
            self.dict[key][self.pos()] = th.tensor(value).to(self.device)
        self.position += 1

    def append(self, other: dict):
        """
        Appends values from another ReplayBuffer or dict to this ReplayBuffer.

        Args:
            other (ReplayBuffer or dict): The ReplayBuffer or dict from which to append values.
        """
        if isinstance(other, ReplayBuffer):
            other = other.dict
        n = None
        for key, value in other.items():
            self.add_value(key, value)
            if n is None:
                n = len(value)
            else:
                assert n == len(value), 'Cannot append ReplayBuffer with different lengths'
        self.position += n

    def sample(self, batch_size=None, device='cpu'):
        """
        Randomly samples experiences from the buffer. If a batch_size was provided when initializing the buffer and no batch_size is provided here, the initial batch_size is used.

        Args:
            batch_size (int, optional): The number of experiences to sample from the buffer.

        Returns:
            ReplayBuffer: A new ReplayBuffer containing the sampled experiences.
        """
        if batch_size is None:
            batch_size = self.batch_size
        idx = th.randperm(len(self))[:batch_size]
        sampled = ReplayBuffer(buffer_size=batch_size, shapes=self.shapes, batch_size=batch_size)
        sampled.append({key: value[idx] for key, value in self.dict.items()})
        return sampled.to_device(device)

    def trim(self):
        """
        Trims the buffer to its current size.

        Returns:
            ReplayBuffer: The trimmed ReplayBuffer.
        """
        for key, value in self.dict.items():
            self.dict[key] = value[:self.__len__()]
        return self

    def save_to_file(self, path):
        """
        Saves the ReplayBuffer to a file.

        Args:
            path (str): The path to the file where the ReplayBuffer should be saved.
        """
        dict_to_np = {key: self.value_to_list(key) for key in self.dict.keys()}
        json.dump(dict_to_np, open(path, 'w'))

    def to_device(self, device):
        """
        Moves the ReplayBuffer to a device.

        Args:
            device (str): The device to move the ReplayBuffer to.
        """
        for key, value in self.dict.items():
            self.dict[key] = value.to(device)
        return self