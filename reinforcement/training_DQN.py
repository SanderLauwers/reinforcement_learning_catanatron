import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym

from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from stable_baselines3.common.callbacks import CheckpointCallback 
from stable_baselines3 import DQN

from reward_func import reward_function

env = gym.make(
	"catanatron_gym:catanatron-v1",
	config={
		"invalid_action_reward": -69,	
		"map_type": "BASE",
		"vps_to_win": 10,
		"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
		"reward_function": reward_function,
		"representation": "vector"
	},
)

STEPS = 1_000_000
LEARNING_RATE = 0.0001
NAME = "DQN__steps="+"{:.0e}".format(STEPS)+"__lrate="+"{:.1e}".format(LEARNING_RATE)
print(NAME)


checkpoint_callback = CheckpointCallback(
  save_freq=2_000_000,
  save_path="./own/reinforcement/models/checkpoint/" + NAME,
  name_prefix=NAME
)

model = DQN("MlpPolicy", env, verbose=0, learning_rate=LEARNING_RATE, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")
# print(model.policy)
model.learn(total_timesteps=STEPS, tb_log_name=NAME, progress_bar=False, callback=checkpoint_callback)

model.save("./own/reinforcement/models/finished/" + NAME + ".zip")

observation, info = env.reset()

# testy bits
# print("sample action:", env.action_space)
# print("observation space shape", env.observation_space.shape)
# print("observation", type(observation))

for _ in range(1):
	env.reset()
	print(env.game)
	done = False

	while not done:
		action = model.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action)

	
	observation, info = env.reset()
	# print(observation)
	print(observation)
	print(info)

env.close()