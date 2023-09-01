import gym  # pip install gym
import numpy as np
import tensorflow

from keras import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
#from core import fit
from rl.agents import DQNAgent  # pip install keras-rl2
from rl.policy import EpsGreedyQPolicy  # important to have gym==0.25.2
from rl.memory import SequentialMemory

env = gym.make("InvertedPendulum-v4") #, render_mode = 'human')  # no render mode to prevent display while training

states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
model = Sequential()
model.add(Flatten(input_shape=(1,states,)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))
    
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=EpsGreedyQPolicy(),  
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(optimizer=Adam(learning_rate=0.001))

agent.fit(env, nb_steps=10000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()