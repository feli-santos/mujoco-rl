# Importar sua classe DQNAgent (ou apropriada) e outras funções necessárias

# Função para renderizar o agente em um ambiente
def render_agent(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=1234)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            env.render()  # Renderiza o ambiente

            total_reward += reward
            obs = next_obs
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
