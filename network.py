from typing import Tuple

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, config: dict):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        # Get hidden layer sizes from config
        hidden_space1, hidden_space2 = config.get("hidden_layers", (32, 64))

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class DQNNetwork(nn.Module):
    """Network for the DQN agent.

    Attributes:
        network (nn.Sequential): A sequential container of modules where the input is passed through each module
                                in order to generate the Q-values for each action.

    Args:
        obs_space_dims (int): The dimensionality of the observation space. Represents the size of the state input.
        action_space_dims (int): The number of possible actions that the agent can take.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, config: dict):
        super().__init__()

        # Get hidden layer sizes from config
        hidden_space1, hidden_space2 = config.get("hidden_layers", (32, 64))

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: A tensor of Q-values corresponding to the actions available in the environment.
        """
        return self.network(state.float())


class PPONetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, config: dict):
        """
        Proximal Policy Optimization Network class.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()

        # Get hidden layer sizes from config
        hidden_space1, hidden_space2 = config.get("hidden_layers", (32, 64))

        # Shared layers for both policy and value heads
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),  # Linear layer with 64 units
            nn.ReLU(),  # ReLU activation function
            nn.Linear(hidden_space1, hidden_space2),  # Linear layer with 32 units
            nn.ReLU(),  # ReLU activation function
        )

        # Policy heads for mean and standard deviation
        self.mean_head = nn.Linear(hidden_space2, action_space_dims)
        self.std_head = nn.Linear(hidden_space2, action_space_dims)

        # Value head for state value estimation
        self.value_head = nn.Linear(hidden_space2, 1)

    def forward_policy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the policy network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            mean (torch.Tensor): Mean of the action distribution.
            std (torch.Tensor): Standard deviation of the action distribution.
        """
        shared_features = self.shared_layers(
            state.float()
        )  # Pass state through shared layers
        mean = self.mean_head(shared_features)  # Compute mean through mean head
        std = torch.log(
            1 + torch.exp(self.std_head(shared_features))
        )  # Compute std through std head
        return mean, std

    def forward_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the value network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            value (torch.Tensor): Estimated value of the input state.
        """
        shared_features = self.shared_layers(
            state.float()
        )  # Pass state through shared layers
        value = self.value_head(shared_features)  # Compute value through value head
        return value
