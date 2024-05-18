import numpy as np

def mask_function(env):
	mask = np.zeros(env.action_space.n)
	try:
		valid = env.unwrapped.get_valid_actions()
		for v in valid: mask[v] = 1
	except:
		return np.ones(len(mask))
	
	return mask

def direct_mask_function(valid_actions, size):
	mask = np.zeros(size)
	try:
		for v in valid_actions: mask[v] = 1
	except:
		return np.ones(len(mask))
	
	return mask