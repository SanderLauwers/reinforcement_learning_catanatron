import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
from gymnasium import spaces
import torch
import time

from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import *

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker

from maskable_DQN import MaskableDQN, MaskableDQNPolicy
from reward_func import reward_function, VP_only_reward_function
from mask_func import mask_function


# env = gym.make(
# 	"catanatron_gym:catanatron-v1",
# 	config={
# 		"invalid_action_reward": -69,	
# 		"map_type": "BASE",
# 		"vps_to_win": 10,
# 		"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
# 		"reward_function": reward_function,
# 		"representation": "vector"
# 	},
# )

BASED = False
PATH = "own/reinforcement/models/checkpoint/t1716016491__MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0.1__af=tanh__opt=Adam__rf=vp__df=0.99__hl=2/t1716016491__MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0.1__af=tanh__opt=Adam__rf=vp__df=0.99__hl=2_12000000_steps.zip"

STEPS = 100_000_000
LEARNING_RATE = 0.0001 # 0.01 seems too high, between 0.00001 and 0.001 seems to be fine-ish
EXPL_FRAC = 0.1
ACT_FN = "lkyrelu"
W_DECAY = 0.1
OPTIMIZER = "RMSP" # TODO: also try with regular SGD?
TRAIN_FREQ = 12
BATCH_SIZE = 32
REWARD_FUNCTION = "vp"
DISCOUNT_FACTOR = 0.99
HIDDEN_LAYERS = 2

NAME = "t{0}__".format(int(time.time())) + "MaskableDQN__steps="+"{:.0e}".format(STEPS)+"__lr="+"{:.1e}".format(LEARNING_RATE) + "__wd={0}".format(W_DECAY) + "__af=" + ACT_FN + "__opt=" + OPTIMIZER + "__rf=" + REWARD_FUNCTION + "__df=" + str(DISCOUNT_FACTOR) + "__hl=" + str(HIDDEN_LAYERS) + ("__BASED" if BASED else "")
print(NAME)

act_fn = torch.nn.modules.activation.Tanh if ACT_FN=="tanh" else torch.nn.modules.activation.ReLU if ACT_FN == "relu" else torch.nn.modules.activation.LeakyReLU if ACT_FN == "lkyrelu" else quit()
opt = torch.optim.Adam if OPTIMIZER == "Adam" else torch.optim.RMSprop if OPTIMIZER == "RMSP" else torch.optim.SGD if OPTIMIZER == "SGD" else quit()
rew_fn = reward_function if REWARD_FUNCTION == "all" else VP_only_reward_function if REWARD_FUNCTION == "vp" else quit()
net_arch = [64 for _ in range(HIDDEN_LAYERS)]

env = CatanatronEnv({
	"invalid_action_reward": -69,	
	"map_type": "BASE",
	"vps_to_win": 10,
	"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
	"reward_function": rew_fn,
	"representation": "vector"
})

env = ActionMasker(env, mask_function)


checkpoint_callback = CheckpointCallback(
  save_freq=2_000_000,
  save_path="./own/reinforcement/models/checkpoint/" + NAME,
  name_prefix=NAME
)

# model = DQN(getMaskableDQNPolicy(mask_function, env), env, learning_rate=LEARNING_RATE, exploration_fraction=0.01, verbose=0, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")
# model = MaskableDQN("MaskableDQNPolicy", env, learning_starts=1000, learning_rate=LEARNING_RATE, exploration_fraction=0.1, verbose=0, tensorboard_log="own/reinforcement/tensorboard_logs", device="cuda")
model = MaskableDQN("MaskableDQNPolicy", env, batch_size=BATCH_SIZE, train_freq=TRAIN_FREQ, gamma=DISCOUNT_FACTOR, learning_starts=1000, learning_rate=LEARNING_RATE, exploration_fraction=EXPL_FRAC,
					policy_kwargs={"optimizer_class": opt, "net_arch": net_arch, "activation_fn": act_fn, "optimizer_kwargs": {"weight_decay": W_DECAY}}, verbose=0, device="cuda", tensorboard_log="own/reinforcement/tensorboard_logs"
					)

if BASED: model.set_parameters(PATH)

model.learn(total_timesteps=STEPS, progress_bar=False, callback=checkpoint_callback, tb_log_name=NAME
			)

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