from catanatron import Player, Color
from catanatron_experimental.cli.cli_players import register_player
from catanatron.game import Game
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import from_action_space
from sb3_contrib.common.wrappers import ActionMasker

import sys
sys.path.append("./own")
sys.path.append("./own/reinforcement") # for loaded

from reinforcement.maskable_DQN import MaskableDQN
from reinforcement.mask_func import mask_function
from reinforcement.reward_func import reward_function


import gymnasium as gym

@register_player("OWNREINFORCEMENT")
class OwnReinforcement(Player):
	def __init__(self, color, is_bot=True):
		self.init_model()
		super().__init__(color, is_bot)

	def init_model(self):
		self._mock_env = gym.make(
			"catanatron_gym:catanatron-v1",
			config={
				"invalid_action_reward": -69,	
				"map_type": "BASE",
				"vps_to_win": 10,
				# "enemies": [AlphaBetaPlayer(Color.RED)], # bot player is blue
				"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
				"reward_function": reward_function,
				"representation": "vector"
			},
		)

		self._mock_env = ActionMasker(self._mock_env, mask_function)

		self._model = MaskableDQN.load("./own/reinforcement/models/MaskableDQN_test", policy="MaskableDQNPolicy", env=self._mock_env)


	def decide(self, game: Game, playable_actions):
		"""Should return one of the playable_actions.

		Args:
				game (Game): complete game state. read-only.
				playable_actions (Iterable[Action]): options to choose from
		Return:
				action (Action): Chosen element of playable_actions
		"""
		# ===== YOUR CODE HERE =====
		
		self._mock_env.reset()
		self._mock_env.unwrapped.game = game

		env_action = self._model.predict(self._mock_env.unwrapped._get_observation(), deterministic=True)
		return from_action_space(env_action[0], self._mock_env.game.state.playable_actions)
	
		# ===== END YOUR CODE =====