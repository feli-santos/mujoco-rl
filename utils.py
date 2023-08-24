import json
import os
from datetime import datetime
from turtle import st
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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


def create_results_folder():
    """Creates a folder to store the results of the experiment.

    Returns:
        The name of the folder
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"experiment_results/{timestamp}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def save_experiment_results(
    results: dict, config: dict, folder_name: Optional[str] = None
):
    """Saves the results of the experiment in a JSON file.

    Args:
        results: The results of the experiment
        config: The config used for the experiment
        folder_name: The name of the folder to store the results
    """
    # Create folder if it doesn't exist
    if folder_name is None or not os.path.exists(folder_name):
        folder_name = create_results_folder()

    # Create a human-readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Merge config and results
    experiment_data = {
        "config": config,
        "results": results,
    }

    # Create JSON file to save the data
    with open(f"{folder_name}/experiment_{timestamp}.json", "w") as f:
        json.dump(experiment_data, f, default=default_serialize, indent=4)


def default_serialize(obj):
    """Default JSON serializer.

    Args:
        obj: Object to serialize

    Returns:
        The serialized object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type '{obj.__class__.__name__}' is not JSON serializable"
    )
