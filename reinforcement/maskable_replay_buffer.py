from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class MaskableReplayBufferSamples(ReplayBufferSamples):
	next_mask: torch.Tensor

class MaskableReplayBuffer(ReplayBuffer):
	next_mask: np.ndarray
	
	def __init__(
		self,
		buffer_size: int,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		device: Union[torch.device, str] = "auto",
		n_envs: int = 1,
		optimize_memory_usage: bool = False,
		handle_timeout_termination: bool = True,
	):
		super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
		self.next_mask = np.zeros((self.buffer_size, action_space.n), dtype=np.int16)

	def add(
			self,
			obs: np.ndarray,
			next_obs: np.ndarray,
			action: np.ndarray,
			reward: np.ndarray,
			done: np.ndarray,
			infos: List[Dict[str, Any]],
		) -> None:
		for info in infos:
			if len(info.get("valid_actions", [])) != 0:
				self.next_mask[self.pos] = np.array(info.get("valid_actions"))

		super().add(obs, next_obs, action, reward, done, infos)

	def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableReplayBufferSamples:
		unmaskable_samples = super()._get_samples(batch_inds, env)
		maskable_samples = MaskableReplayBufferSamples(unmaskable_samples.observations, unmaskable_samples.actions, unmaskable_samples.next_observations, unmaskable_samples.dones, unmaskable_samples.rewards)
		maskable_samples.next_mask = self.to_torch(self.next_mask[batch_inds])
		return maskable_samples
		