import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import numpy as np
import time

from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from reward_func import reward_function
from mask_func import mask_function

env = gym.make(
	"catanatron_gym:catanatron-v1",
	config={
		"invalid_action_reward": -1,	
		"map_type": "BASE",
		"vps_to_win": 10,
		"enemies": [AlphaBetaPlayer(Color.RED)], # bot player is blue
		"reward_function": reward_function,
		"representation": "vector"
	},
)

env = ActionMasker(env, mask_function)  

STEPS = 10_000_000
NAME = "MaskablePPO__steps="+"{:.0e}".format(STEPS) + "__{0}".format(int(time.time()))
print(NAME)

checkpoint_callback = CheckpointCallback(
  save_freq=1_000_000,
  save_path="./own/reinforcement/models/checkpoint/" + NAME,
  name_prefix=NAME
)

model = MaskablePPO("MlpPolicy", env, verbose=0, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")

model.learn(total_timesteps=STEPS, tb_log_name=NAME, callback=checkpoint_callback)

model.save("./own/reinforcement/models/finished/" + NAME + ".zip")

env.close()