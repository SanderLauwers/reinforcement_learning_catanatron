import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import time

from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker

from maskable_DQN import MaskableDQN
from reward_func import reward_function
from mask_func import mask_function


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

env = ActionMasker(env, mask_function)

STEPS = 100_000_000
LEARNING_RATE = 0.00001
NAME = "MaskableDQN__steps="+"{:.0e}".format(STEPS)+"__lrate="+"{:.1e}".format(LEARNING_RATE) + "__{0}".format(int(time.time()))
print(NAME)

checkpoint_callback = CheckpointCallback(
  save_freq=2_000_000,
  save_path="./own/reinforcement/models/checkpoint/" + NAME,
  name_prefix=NAME
)

# default learning_rate-0.0001, train_freq=4
# TODO: params learning rate
# model = DQN(getMaskableDQNPolicy(mask_function, env), env, learning_rate=LEARNING_RATE, exploration_fraction=0.01, verbose=0, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")
model = MaskableDQN("MaskableDQNPolicy", env, learning_starts=1000, learning_rate=LEARNING_RATE, exploration_fraction=0.025, verbose=0, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")

model.learn(total_timesteps=STEPS, tb_log_name=NAME, progress_bar=False, callback=checkpoint_callback)

model.save("./own/reinforcement/models/finished/" + NAME + ".zip")

observation, info = env.reset()

for _ in range(10):
	observation, info = env.reset()
	done = False

	step = 0
	cum_reward = 0
	while not done:
		action = model.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action[0])
		cum_reward += reward
		if reward == -69:
			print("invalid action")
		if terminated or truncated or step >= 1069:
			done = True
		step += 1

	print(str(_+1) + ":", "WON!" if reward == 1 else "LOST!" if reward == -1 else "Invalid action limit!" if reward == -69 else "DRAW?/TIMEOUT", "In " + str(step) + " timesteps, with a cumulative reward of " + str(cum_reward) + ".")

env.close()