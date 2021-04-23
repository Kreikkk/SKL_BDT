import numpy as np
from sys import argv
import pickle

from dataloader import dataset_gen

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from plotters import dump
import time


def train(TrainDF, params, filename="output"):
	train_data = np.array(TrainDF.iloc[:,:11], dtype="float64")
	labels = np.array(TrainDF.iloc[:,14], dtype="float64")

	tree = DecisionTreeClassifier()
	grad = GradientBoostingClassifier(n_estimators=int(params[0]),
									  max_depth=float(params[1]),
									  learning_rate=float(params[2]),
									  random_state=1,
									  verbose=1)
	t = time.time()
	grad.fit(train_data, labels)
	dump(filename, str(round(time.time() - t, 3)))

	with open(f"models/{filename}.pickle", "wb") as file:
		pickle.dump(grad, file)


if __name__ == "__main__":
	filename = argv[1]
	params = argv[2:]
	TrainDF, TestDF = dataset_gen(backgrounds="train")

	train(TrainDF, params, filename=filename)