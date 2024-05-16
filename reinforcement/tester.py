import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import numpy as np
import random

from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron import Player
from catanatron_gym.envs.catanatron_env import *

from reward_func import reward_function
from mask_func import mask_function
from maskable_DQN import MaskableDQN

from stable_baselines3 import DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

PATH = "own/reinforcement/models/checkpoint/MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0__af=tanh__opt=Adam__1715719440/MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0__af=tanh__opt=Adam__1715719440_4000000_steps.zip"

class OwnRandomPlayer(Player):
  def decide(self, game, playable_actions):
    return playable_actions[random.randint(0, len(playable_actions) - 1)]
  
env = CatanatronEnv({
	"invalid_action_reward": -69,	
	"map_type": "BASE",
	"vps_to_win": 10,
	"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
	"reward_function": reward_function,
	"representation": "vector"
})

env = ActionMasker(env, mask_function)

# model = MaskablePPO("MlpPolicy", env, verbose=1)
# model.set_parameters("./own/reinforcement/models/masked_ppo_catanatron5_000_000")

# model = DQN("MlpPolicy", env, verbose=1)
# model.set_parameters("./own/reinforcement/models/dqn_catanatron10000000")

model = MaskableDQN.load(PATH, policy="MaskableDQNPolicy", env=env)

won_amount = 0
lost_amount = 0
for _ in range(100):
	observation, info = env.reset()
	done = False

	step = 0
	cum_reward = 0
	while not done:
		# action = model.predict(observation, deterministic=True, action_masks=mask_function(env))
		action = model.predict(observation, deterministic=True)
		observation, reward, terminated, truncated, info = env.step(action[0])

		cum_reward += reward
		step += 1

		if reward == -69:
			print("invalid action1")
		if terminated or truncated or step >= 2000:
			done = True

	if reward == 1: won_amount += 1
	elif reward == -1: lost_amount += 1
	print(str(_+1) + ":", "WON!" if reward == 1 else "LOST!" if reward == -1 else "Invalid action limit!" if reward == -69 else "DRAW?/TIMEOUT", "In " + str(step) + " timesteps, with a cumulative reward of " + str(cum_reward) + ".")

print("Won", won_amount, "times and lost", lost_amount, "times.")

env.close()