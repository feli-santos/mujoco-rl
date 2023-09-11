import gymnasium as gym

def create_env(flag: bool = True) -> tuple[any, int, int]:
    """Creates and wraps the environment for training

    Returns:
        env: Wrapped environment
    """
    if flag is True:
        env = gym.make("InvertedPendulum-v4") #, render_mode='human')
          # Records episode-reward
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # type: ignore

        # Observation-space of InvertedPendulum-v4
        obs_space_dims = env.observation_space.shape[0] # type: ignore

        # Action-space of InvertedPendulum-v4
        action_space_dims = env.action_space.shape[0] # type: ignore
    else:
        env = gym.make("CartPole-v1") #, render_mode = 'human')
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        obs_space_dims = env.observation_space.shape[0]
        action_space_dims = int(env.action_space.n)

    return wrapped_env, obs_space_dims, action_space_dims
