import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_learning_curve(rewards_over_seeds: list[list[int]]):
    """Plots the learning curve based on the rewards over seeds.

    Args:
        rewards_over_seeds: List of rewards for each seed
    """
    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds] # type: ignore
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    plt.show()