import numpy as np
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn import DQN
from sb3_contrib.common.wrappers import ActionMasker
import torch
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from sb3_contrib.common.maskable.utils import get_action_masks
from gymnasium import spaces
from stable_baselines3.common.noise import ActionNoise


class MaskableQNetwork(QNetwork):
	def _predict(self, observation: PyTorchObs, action_masks: np.ndarray, deterministic: bool = True) -> torch.Tensor:
		q_values = self(observation)

		# Get largest Q from masked actions
		masks_tensor = torch.from_numpy(action_masks < 1).to(device=self.device)#.cuda("cuda:0")
		large = torch.finfo(q_values.dtype).max
		action = (q_values - large * masks_tensor - large * masks_tensor).argmax(dim=1).reshape(-1)

		return action

	
class MaskableDQNPolicy(DQNPolicy):
	q_net: MaskableQNetwork
	q_net_target: MaskableQNetwork

	def make_q_net(self) -> MaskableQNetwork:
		net_args = self._update_features_extractor(self.net_args, features_extractor=None)
		return MaskableQNetwork(**net_args).to(self.device)
	
	def _predict(self, obs: PyTorchObs, action_masks: np.ndarray, deterministic: bool = True) -> torch.Tensor:
		return self.q_net._predict(obs, action_masks, deterministic=deterministic)

	def predict(
			self,
			observation: Union[np.ndarray, Dict[str, np.ndarray]],
			action_masks: np.ndarray,
			state: Optional[Tuple[np.ndarray, ...]] = None,
			episode_start: Optional[np.ndarray] = None,
			deterministic: bool = False,
		) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
			"""
			Get the policy action from an observation (and optional hidden state).
			Includes sugar-coating to handle different observations (e.g. normalizing images).

			:param observation: the input observation
			:param state: The last hidden states (can be None, used in recurrent policies)
			:param episode_start: The last masks (can be None, used in recurrent policies)
				this correspond to beginning of episodes,
				where the hidden states of the RNN must be reset.
			:param deterministic: Whether or not to return deterministic actions.
			:return: the model's action and the next hidden state
				(used in recurrent policies)
			"""
			# Switch to eval mode (this affects batch norm / dropout)
			self.set_training_mode(False)

			# Check for common mistake that the user does not mix Gym/VecEnv API
			# Tuple obs are not supported by SB3, so we can safely do that check
			if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
				raise ValueError(
					"You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
					"You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
					"vs `obs = vec_env.reset()` (SB3 VecEnv). "
					"See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
					"and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
				)

			obs_tensor, vectorized_env = self.obs_to_tensor(observation)

			with torch.no_grad():
				actions = self._predict(obs_tensor, action_masks, deterministic=deterministic)
			# Convert to numpy, and reshape to the original action shape
			actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

			if isinstance(self.action_space, spaces.Box):
				if self.squash_output:
					# Rescale to proper domain when using squashing
					actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
				else:
					# Actions could be on arbitrary scale, so clip the actions to avoid
					# out of bound error (e.g. if sampling from a Gaussian distribution)
					actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

			# Remove batch dimension if needed
			if not vectorized_env:
				assert isinstance(actions, np.ndarray)
				actions = actions.squeeze(axis=0)

			return actions, state  # type: ignore[return-value]
	

class MaskableDQN(DQN):
	def __init__(
	self,
	policy: Union[str, Type[DQNPolicy]],
	env: Union[ActionMasker, str],
	learning_rate: Union[float, Schedule] = 1e-4,
	buffer_size: int = 1_000_000,  # 1e6
	learning_starts: int = 100,
	batch_size: int = 32,
	tau: float = 1.0,
	gamma: float = 0.99,
	train_freq: Union[int, Tuple[int, str]] = 4,
	gradient_steps: int = 1,
	replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
	replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
	optimize_memory_usage: bool = False,
	target_update_interval: int = 10000,
	exploration_fraction: float = 0.1,
	exploration_initial_eps: float = 1.0,
	exploration_final_eps: float = 0.05,
	max_grad_norm: float = 10,
	stats_window_size: int = 100,
	tensorboard_log: Optional[str] = None,
	policy_kwargs: Optional[Dict[str, Any]] = None,
	verbose: int = 0,
	seed: Optional[int] = None,
	device: Union[torch.device, str] = "auto",
	_init_setup_model: bool = True,
) -> None:
		# FIXME: stupid workaround; why doesn't self.env work?
		self.___env = env
		if (policy == "MaskableDQNPolicy"):
			policy = MaskableDQNPolicy
		super().__init__(policy,
				   env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq,
				   gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, target_update_interval,
				   exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, stats_window_size,
				   tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)
	
	policy: MaskableDQNPolicy

	def predict(
		self,
		observation: Union[np.ndarray, Dict[str, np.ndarray]],
		state: Optional[Tuple[np.ndarray, ...]] = None,
		episode_start: Optional[np.ndarray] = None,
		deterministic: bool = False,
	) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
		action_masks = get_action_masks(self.env)
		if not deterministic and np.random.rand() < self.exploration_rate:
			if self.policy.is_vectorized_observation(observation):
				if isinstance(observation, dict):
					n_batch = observation[next(iter(observation.keys()))].shape[0]
				else:
					n_batch = observation.shape[0]
				action = np.array([self.action_space.sample(action_masks[_].astype(np.int8)) for _ in range(n_batch)])
			else:
				action = np.array(self.action_space.sample(action_masks[0].astype(np.int8)))
		else:
			action, state = self.policy.predict(observation, action_masks, state, episode_start, deterministic)
		return action, state
	
	def _sample_action(
		self,
		learning_starts: int,
		action_noise: Optional[ActionNoise] = None,
		n_envs: int = 1,
	) -> Tuple[np.ndarray, np.ndarray]:
		# Select action randomly or according to policy
		if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
			# Warmup phase, changed for maskable
			unscaled_action = np.array([self.action_space.sample(get_action_masks(self.env)[_].astype(np.int8)) for _ in range(n_envs)])
		else:
			assert self._last_obs is not None, "self._last_obs was not set"
			unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

		# Rescale the action from [low, high] to [-1, 1]
		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)

			# Add noise to the action (improve exploration)
			if action_noise is not None:
				scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

			# We store the scaled action in the buffer
			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			# Discrete case, no need to normalize or clip
			buffer_action = unscaled_action
			action = buffer_action
		return action, buffer_action