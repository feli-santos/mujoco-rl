import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns


def plot_learning_curve(rewards_over_seeds: dict, title: str):
    """Plots the learning curve based on the rewards over seeds.

    Args:
        rewards_over_seeds: Dictionary where keys are algorithm names and values are lists of rewards for each seed
        title: The title of the plot
    """
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    for algo_name, rewards in rewards_over_seeds.items():
        rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards]
        df = pd.DataFrame(rewards_to_plot).melt()
        df.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
        sns.lineplot(x="episodes", y="reward", data=df, label=algo_name).set(
            title=title
        )

    plt.legend()
    plt.show(block=False)
    plt.pause(10)  # plot will be displayed for 10 seconds


def read_config(json_file_path: str):
    """Reads the config file.

    Args:
        json_file_path: The path to the config file

    Returns:
        The config file as a dictionary
    """
    with open(json_file_path, "r") as file:
        config = json.load(file)
    return config
