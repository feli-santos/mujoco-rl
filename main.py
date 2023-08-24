from train import train_dqn, train_ppo, train_reinforce
from utils import plot_learning_curve, read_config

if __name__ == "__main__":
    num_episodes = 1000
    random_seeds = [1234, 5678, 9012]

    # Train agents
    # reinforce_rewards = train_reinforce(num_episodes, random_seeds)
    # dqn_rewards = train_dqn(num_episodes, random_seeds)
    ppo_rewards = train_ppo(num_episodes, random_seeds)

    rewards_over_seeds = {
        # "REINFORCE": reinforce_rewards,
        # "DQN": dqn_rewards,
        "PPO": ppo_rewards,
    }

plot_learning_curve(
    rewards_over_seeds, "REINFORCE vs DQN vs PPO for InvertedPendulum-v4"  # type: ignore
)
