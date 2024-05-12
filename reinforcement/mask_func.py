import numpy as np

def mask_function(env):
	mask = np.zeros(env.action_space.n)
	try:
		valid = env.unwrapped.get_valid_actions()
		for v in valid:
			mask[v] = 1
	except:
		return np.ones(len(mask))
	
	return mask