from catanatron import Player, Color
from catanatron_experimental.cli.cli_players import register_player
from catanatron.game import Game, Action
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import *
from sb3_contrib.common.wrappers import ActionMasker

import torch
from typing import Iterable
import gymnasium as gym
from gymnasium import spaces

import sys
sys.path.append("./own")
sys.path.append("./own/reinforcement") # for loaded

from reinforcement.maskable_DQN import MaskableDQN
from reinforcement.mask_func import mask_function
from reinforcement.reward_func import reward_function

PATH = "own/reinforcement/models/checkpoint/t1715889606__MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0.1__af=tanh__opt=Adam__rf=all__df=0.99__hl=1/t1715889606__MaskableDQN__steps=1e+08__lr=1.0e-04__wd=0.1__af=tanh__opt=Adam__rf=all__df=0.99__hl=1_54000000_steps.zip"
@register_player("OWNREINFORCEMENT")
class OwnReinforcement(Player):
	i = 0
	def __init__(self, color, is_bot=True):
		self.init_model()
		super().__init__(color, is_bot)

	def init_model(self):
		self._mock_env = CatanatronEnv({
			"invalid_action_reward": -69,	
			"map_type": "BASE",
			"vps_to_win": 10,
			"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
			"reward_function": reward_function,
			"representation": "vector"
		})

		# print(self._mock_env.observation_space)

		self._mock_env = ActionMasker(self._mock_env, mask_function)

		self._model = MaskableDQN.load(PATH, policy="MaskableDQNPolicy", env=self._mock_env)
		print(self._model.policy)
		print("model_initiated")


	def decide(self, game: Game, playable_actions: Iterable[Action]):
		"""Should return one of the playable_actions.

		Args:
				game (Game): complete game state. read-only.
				playable_actions (Iterable[Action]): options to choose from
		Return:
				action (Action): Chosen element of playable_actions
		"""

		if len(playable_actions) == 1: return playable_actions[0]

		self._mock_env.unwrapped.players = self._mock_env.unwrapped.game.state.players
		self._mock_env.unwrapped.game = game

		env_action = self._model.predict(self._mock_env.unwrapped._get_observation(), deterministic=True)

		# print("action:")
		# print([p.action_type for p in playable_actions])
		# print(self._mock_env.unwrapped.get_valid_actions())
		# print([from_action_space(v, playable_actions).action_type for v in self._mock_env.unwrapped.get_valid_actions()])
		# print(mask_function(self._mock_env))
		# print(env_action[0], from_action_space(env_action[0], playable_actions))
		# print()
		return from_action_space(env_action[0], playable_actions)
	
		# ===== END YOUR CODE =====