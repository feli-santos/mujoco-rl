import numpy as np
from tqdm import tqdm
import random
import torch

from environment import create_env
from agent import REINFORCE, DQNAgent

from typing import Type


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
                )  # needs implementation
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
