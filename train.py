import random

import numpy as np
import torch
from tqdm import tqdm

from agent import REINFORCE, DQNAgent, PPOAgent
from environment import create_env


def train_reinforce(num_episodes: int, random_seeds: list[int]) -> list[list[int]]:
    """Trains the policy using REINFORCE algorithm with different random seeds.

    Args:
        num_episodes: Total number of episodes
        random_seeds: List of random seeds for training

    Returns:
        rewards_over_seeds: List of rewards for each seed
    """
    rewards_over_seeds = []

    for seed in tqdm(random_seeds):
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create environment
        wrapped_env, obs_space_dims, action_space_dims = create_env()

        # Reinitialize agent for each seed
        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        for episode in tqdm(range(num_episodes)):
            obs, _ = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.choose_action(obs)  # type: ignore

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, _ = wrapped_env.step(action)  # type: ignore
                agent.rewards.append(reward)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            # Print average reward every 100 episodes
            if episode % 100 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))  # type: ignore
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    return rewards_over_seeds


def train_dqn(num_episodes: int, random_seeds: list[int]) -> list[list[int]]:
    """Trains the DQN agent with different random seeds.

    Args:
        num_episodes: Total number of episodes
        random_seeds: List of random seeds for training

    Returns:
        rewards_over_seeds: List of rewards for each seed
    """
    rewards_over_seeds = []

    for seed in tqdm(random_seeds):
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create environment
        wrapped_env, obs_space_dims, action_space_dims = create_env()

        # Reinitialize agent for each seed
        agent = DQNAgent(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        for episode in tqdm(range(num_episodes)):
            obs, _ = wrapped_env.reset(seed=seed)
            done = False
            while not done:
                action = agent.choose_action(obs)
                next_obs, reward, terminated, truncated, _ = wrapped_env.step(action)  # type: ignore

                # store transition and learn
                agent.store_transition(
                    obs, action, reward, next_obs, terminated or truncated
                )
                agent.learn()

                # End the episode when either truncated or terminated is true
                done = terminated or truncated

                # Update the observation
                obs = next_obs

            reward_over_episodes.append(wrapped_env.return_queue[-1])

            # update target network
            if episode % agent.update_freq == 0:
                agent.update_target_network()

            # Print average reward every 100 episodes
            if episode % 100 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))  # type: ignore
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    return rewards_over_seeds


def train_ppo(num_episodes: int, random_seeds: list[int]) -> list[list[int]]:
    """Trains the PPO agent with different random seeds.

    Args:
        num_episodes: Total number of episodes
        random_seeds: List of random seeds for training

    Returns:
        rewards_over_seeds: List of rewards for each seed
    """

    # List to store rewards for different random seeds
    rewards_over_seeds = []

    # Iterate through each random seed using tqdm for a progress bar
    for seed in tqdm(random_seeds):
        # Set the random seed for PyTorch, random, and NumPy for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create the environment and get observation and action dimensions
        wrapped_env, obs_space_dims, action_space_dims = create_env()

        # Initialize the PPO agent with the observation and action dimensions
        agent = PPOAgent(obs_space_dims, action_space_dims)

        # List to store rewards for each episode in this random seed
        reward_over_episodes = []

        # Iterate through each episode
        for episode in tqdm(range(num_episodes)):
            # Reset the environment to the initial state and obtain the first observation
            obs, _ = wrapped_env.reset(seed=seed)

            # Boolean flag to track if the episode has terminated
            done = False

            # Execute steps within the episode until termination
            while not done:
                # Use the agent to choose an action given the current observation
                action = agent.choose_action(obs)

                # Take a step in the environment using the chosen action and obtain the next observation, reward, etc.
                next_obs, reward, terminated, truncated, _ = wrapped_env.step(action)

                # Compute the advantage and old log probability
                advantage, old_log_prob = agent.compute_advantage_and_log_prob(
                    obs, action, reward, next_obs, terminated or truncated
                )

                # Store the transition in the agent's memory (e.g. for use in a replay buffer)
                agent.store_transition(
                    obs,
                    action,
                    reward,
                    next_obs,
                    terminated or truncated,
                    advantage,
                    old_log_prob,
                )

                # Update the current observation to the next observation
                obs = next_obs

                # Check for termination conditions (e.g., max steps reached)
                if terminated or truncated:
                    # Update the agent's policy and value functions
                    agent.update()

                    # Set the termination flag to exit the loop
                    done = True

            # Append the total reward for this episode to the reward list
            reward_over_episodes.append(wrapped_env.return_queue[-1])

        # Append the reward list for this seed to the overall list
        rewards_over_seeds.append(reward_over_episodes)

    # Return the final list of rewards, organized by random seed and episode
    return rewards_over_seeds
