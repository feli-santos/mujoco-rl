import random
from collections import deque

import numpy as np
import torch
from torch.distributions.normal import Normal

from network import DQNNetwork, PolicyNetwork, PPONetwork


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, config: dict):
        """Initializes an agent that learns a policy via REINFORCE algorithm
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # config
        self.learning_rate = config.get("learning_rate", 1e-4)  # Learning rate
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.eps = config.get("eps", 1e-6)  # Epsilon value for the normal distribution

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNetwork(obs_space_dims, action_space_dims, config)
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

    def __init__(self, obs_space_dims: int, action_space_dims: int, config: dict):
        self.memory = deque(maxlen=config.get("memory_size", 10000))  # memory buffer
        self.gamma = config.get("gamma", 0.99)  # discount factor
        self.epsilon = config.get("epsilon", 1.0)  # exploration rate
        self.epsilon_min = config.get("epsilon_min", 0.01)  # minimum exploration rate
        self.epsilon_decay = config.get(
            "epsilon_decay", 0.995
        )  # decay rate for exploration rate
        self.batch_size = config.get("batch_size", 64)  # batch size
        self.update_freq = config.get("update_freq", 1000)  # update frequency
        self.learning_rate = config.get("learning_rate", 0.001)  # learning rate

        # networks
        self.action_space_dims = action_space_dims
        self.q_network = DQNNetwork(obs_space_dims, action_space_dims, config)
        self.target_network = DQNNetwork(obs_space_dims, action_space_dims, config)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )

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


class PPOAgent:
    """Agent that learns to solve the environment using PPO."""

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        config: dict,
    ):
        """
        Initialize a PPOAgent with given observation and action dimensions.

        :param obs_space_dims: Dimensions of the observation space.
        :param action_space_dims: Dimensions of the action space.
        :param lr: Learning rate for the optimizer (default is 0.0003).
        :param clip_epsilon: Epsilon value for the clipping in the objective function (default is 0.2).
        :param update_epochs: Number of epochs for the update step (default is 10).
        """
        self.obs_space_dims = obs_space_dims  # Observation space dimensions
        self.action_space_dims = action_space_dims  # Action space dimensions

        self.policy_net = PPONetwork(
            obs_space_dims, action_space_dims, config
        )  # Neural network for the policy

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.get("learning_rate", 0.0003)
        )  # Optimizer

        self.clip_epsilon = config.get("clip_epsilon", 0.2)  # Epsilon for clipping
        self.trajectory = []  # Trajectory for the current episode
        self.update_epochs = config.get(
            "update_epochs", 10
        )  # Number of epochs for the update step
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.lam = config.get("lam", 0.95)  # Lambda for GAE

    def choose_action(self, state: np.ndarray) -> float:
        """
        Choose an action based on the current state using the policy network.

        :param state: Current state.
        :return: Sampled action.
        """
        with torch.no_grad():
            state = torch.tensor(np.array([state]))  # type: ignore

            # Using forward_policy
            mean, std = self.policy_net.forward_policy(state)  # type: ignore
            dist = Normal(mean, std)  # Create a normal distribution
            action = dist.sample()  # Sample an action from the distribution
            action = action.squeeze(-1)  # Remove the extra dimension
        return action.numpy()

    def store_transition(
        self, state, action, reward, next_state, done, advantage, old_log_prob
    ):
        """
        Store a transition in the trajectory.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state.
        :param done: Whether the episode is done.
        :param advantage: Advantage of the action.
        :param old_log_prob: Old log probability of the action.
        """
        self.trajectory.append(
            (state, action, reward, next_state, done, advantage, old_log_prob)
        )

    def update(self):
        """
        Update the policy network using the stored trajectory.
        """
        # Unpack the trajectory
        state_arr, action_arr, reward_arr, _, _, advantage_arr, old_log_prob_arr = map(
            lambda x: torch.tensor(x, dtype=torch.float32), zip(*self.trajectory)
        )
        for _ in range(self.update_epochs):  # Update over several epochs
            mean, std = self.policy_net.forward_policy(state_arr)
            dist = Normal(mean, std)  # Create a normal distribution
            log_prob = dist.log_prob(action_arr)
            ratio = torch.exp(log_prob - old_log_prob_arr)
            surr1 = ratio * advantage_arr
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantage_arr
            )
            loss = -torch.min(surr1, surr2).mean()  # PPO loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.trajectory = []  # Clear the trajectory for the next episode

    def compute_advantage_and_log_prob(
        self, states, actions, rewards, next_states, dones
    ):
        states = torch.tensor(states, dtype=torch.float32)  # Convert states to tensors
        next_states = torch.tensor(
            next_states, dtype=torch.float32
        )  # Convert next_states to tensors
        actions = torch.tensor(
            actions, dtype=torch.float32
        )  # Convert actions to tensors

        with torch.no_grad():
            values = self.policy_net.forward_value(states)  # Using forward_value
            next_values = self.policy_net.forward_value(
                next_states
            )  # Using forward_value

            # Compute TD errors
            td_errors = rewards + self.gamma * next_values * (1 - dones) - values

            # Compute advantage using GAE
            advantages = []
            advantage = 0
            for td_error in reversed(td_errors):
                advantage = td_error + self.gamma * self.lam * advantage
                advantages.insert(0, advantage)
            advantages = torch.tensor(advantages)

            # Compute old log probabilities using the policy
            means, stds = self.policy_net.forward_policy(states)
            dist = Normal(means, stds)
            old_log_probs = dist.log_prob(actions)

        return advantages, old_log_probs
