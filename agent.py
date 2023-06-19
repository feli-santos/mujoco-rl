import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from network import PolicyNetwork


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNetwork(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state])) # type: ignore
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward() # type: ignore
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


class DQN:
    """Deep Q-Network (DQN) algorithm implementation."""

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        replay_memory_size: int = 10000,
        batch_size: int = 32,
        update_target_freq: int = 100,
    ):
        """
        Initialize the DQN agent.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            epsilon (float, optional): Exploration rate. Defaults to 1.0.
            epsilon_decay (float, optional): Decay rate for exploration rate. Defaults to 0.995.
            epsilon_min (float, optional): Minimum exploration rate. Defaults to 0.01.
            replay_memory_size (int, optional): Size of the replay memory. Defaults to 10000.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            update_target_freq (int, optional): Frequency of updating the target network. Defaults to 100.
        """
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq

        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def build_q_network(self) -> nn.Module:
        """Build the Q-network architecture."""
        model = nn.Sequential(
            nn.Linear(self.obs_space_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_dims),
        )
        return model

    def update_target_network(self):
        """Update the weights of the target network with the current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition (state, action, reward, next_state, done) in the replay memory.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode terminated after this transition.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    def choose_action(self, state) -> int:
        """
        Select an action based on the current state using an epsilon-greedy policy.

        Args:
            state: Current state.

        Returns:
            Action to take.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space_dims)
        else:
            state = torch.FloatTensor(state)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self):
        """Perform a single training step using a batch of samples from the replay memory."""
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * ~dones

        loss = self.loss_function(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update(self, state, action, reward, next_state, done):
        """
        Update the agent based on the observed transition.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode terminated after this transition.
        """
        self.store_transition(state, action, reward, next_state, done)
        self.train()

        if self.update_target_freq > 0 and len(self.replay_memory) % self.update_target_freq == 0:
            self.update_target_network()
