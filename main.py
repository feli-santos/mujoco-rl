from utils import plot_learning_curve
from train import train_dqn, train_reinforce

if __name__ == "__main__":
    num_episodes = 1000
    random_seeds = [1234]

    # Train agents
    # reinforce_rewards = train_reinforce(num_episodes, random_seeds)
    dqn_rewards = train_dqn(num_episodes, random_seeds)

    rewards_over_seeds = {
        # "REINFORCE": reinforce_rewards,
        "DQN": dqn_rewards,
    }

#plot_learning_curve(rewards_over_seeds, "REINFORCE vs DQN for InvertedPendulum-v4")  # type: ignore
