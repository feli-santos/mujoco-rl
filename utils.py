from numpy import block
from agent import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import torch
import matplotlib
import pickle

def plot_learning_curve(rewards_over_seeds: dict, title: str) -> None:
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

def plot_durations(episode_durations: list[int], show_result: bool =False) -> None:
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display #type: ignore

    plt.ion()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf()) #type: ignore
            display.clear_output(wait=True) #type: ignore
        else:
            display.display(plt.gcf()) #type: ignore

def plot_weight_update(iterations: list[int], average_updates: list[int]) -> None:
    plt.ioff()
    plt.show()
    plt.figure()
    plt.plot(iterations, average_updates, label='Average Weight Update')
    plt.xlabel('Iterations')
    plt.ylabel('Average Update')
    plt.title('Average Weight Update over Iterations')
    plt.legend()
    plt.show()

def save_dqn_agent(agent: DQNAgent, filename: str = 'dqn_agent.pkl') -> None:
    """Salvamento dos pesos do agente

    Parameters
    ----------
    agent : DQNAgent
        Agente treinado
    filename : str, optional
        Nome do arquivo, by default 'dqn_agent.pkl'
    """

    with open(filename, 'wb') as file:
        pickle.dump(agent, file)
    
def load_dqn_agent(filename: str, obs_space_dims: int, action_space_dims: int) -> DQNAgent:
    """Carregamento dos pesos do agente

    Parameters
    ----------
    filename : str
        Nome do arquivo
    obs_space_dims : int
        Número de observações
    action_space_dims : int
        Número de ações

    Returns
    -------
    DQNAgent
        Agente DQN
    """

    # todo considerar as várias seeds para dar load/salvar
    # todo carregar os hiperparâmetros
    
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return DQNAgent(obs_space_dims, action_space_dims)