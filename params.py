from sys import argv


default = {"l": 0.1,
		   "n": 100,
		   "s": 1,
		   "d": 3,
		   "v": 0,}

true_keys = {"learning_rate": 0,
			 "n_estimators":  0,
			 "subsample":	  0,
			 "max_depth":	  0,
			 "verbose":		  0,}

def read_params():
	params = argv[1:]
	if len(params)%2:
		raise KeyError("Wrong arguments format")

	for key, val in zip(params[::2], params[1::2]):
		if key not in default.keys():
			raise KeyError("Wrong key")

		try:
			param = float(val)
		except ValueError:
			default[key] = val
		else:
			default[key] = param

	for key, true_key in zip(default.keys(), true_keys.keys()):
		if key == "n":
			true_keys[true_key] = int(default[key])
			continue
		true_keys[true_key] = default[key]

	return true_keys


if __name__ == "__main__":
	print(read_params())