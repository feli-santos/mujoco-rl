from train import train_dqn, train_ppo, train_reinforce
from utils import (
    create_results_folder,
    plot_learning_curve,
    read_config,
    save_experiment_results,
)

if __name__ == "__main__":
    num_episodes = 1000
    random_seeds = [1234, 5678, 9012]

    # Create results folder
    folder_name = create_results_folder()

    # Read the global config file
    config = read_config("config.json")

    # Train agents
    reinforce_rewards = train_reinforce(num_episodes, random_seeds, config)
    dqn_rewards = train_dqn(num_episodes, random_seeds, config)
    ppo_rewards = train_ppo(num_episodes, random_seeds, config)

    # Store rewards over seeds
    rewards_over_seeds = {
        "REINFORCE": reinforce_rewards,
        "DQN": dqn_rewards,
        "PPO": ppo_rewards,
    }

    # Save experiments
    save_experiment_results(rewards_over_seeds, config, folder_name)

    # Plot learning curve
    plot_learning_curve(
        rewards_over_seeds, "REINFORCE vs DQN vs PPO for InvertedPendulum-v4"  # type: ignore
    )
