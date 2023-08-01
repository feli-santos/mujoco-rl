import random
from collections import deque

import numpy as np
import torch
from torch.distributions.normal import Normal

from network import PolicyNetwork, DQNNetwork


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

    def choose_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))  # type: ignore
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
        loss.backward()  # type: ignore
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


class DQNAgent:
    """Agent that learns to solve the environment using DQN."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.memory = deque(maxlen=10000)  # experience replay
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # minimal exploration rate
        self.epsilon_decay = 0.995  # exploration decay
        self.batch_size = 64  # batch size for the experience replay
        self.update_freq = 1000  # frequency of updating the target network

        # networks
        self.action_space_dims = action_space_dims
        self.q_network = DQNNetwork(obs_space_dims, action_space_dims)
        self.target_network = DQNNetwork(obs_space_dims, action_space_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # Exploration - exploitation trade-off
        if np.random.rand() <= self.epsilon:
            return np.array([random.randrange(self.action_space_dims)])
        else:
            state = torch.tensor(np.array([state]))  # type: ignore
            q_values = self.q_network(state).detach().numpy()
            return np.array([np.argmax(q_values)])

    def learn(self):
        # Make sure the replay buffer is at least batch size large
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(np.array([state]))  # type: ignore
            next_state = torch.tensor(np.array([next_state]))  # type: ignore
            target = reward
            if not done:
                target = (
                    reward
                    + self.gamma * torch.max(self.target_network(next_state)).item()
                )

            current = self.q_network(state)[action]

            target = torch.tensor(
                target, dtype=torch.float32
            )  # convert target to tensor
            loss = torch.nn.functional.mse_loss(current, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
