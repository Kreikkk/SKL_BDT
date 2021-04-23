from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

from dataloader import extract
from params import read_params

import time
import pickle
import os
import json

from plotters import BDT_output_hist_plot, significance_plot


def dataset_gen(ratio=0.5):
	SDataframe, BDataframe = extract()
	SDataframe = SDataframe.sample(frac=1, random_state=1).reset_index(drop=True)
	BDataframe = BDataframe.sample(frac=1, random_state=1).reset_index(drop=True)

	STrainLen, BTrainLen = round(ratio*len(SDataframe)), round(ratio*len(BDataframe))

	STrainDF = SDataframe.iloc[:STrainLen]
	BTrainDF = BDataframe.iloc[:BTrainLen]

	STrainSubset = STrainDF.values
	BTrainSubset = BTrainDF.values

	STestDF = SDataframe.iloc[STrainLen:]
	BTestDF = BDataframe.iloc[BTrainLen:]

	STestSubset = STestDF.values
	BTestSubset = BTestDF.values

	TrainSubset = np.vstack((STrainSubset, BTrainSubset))

	return TrainSubset, STestDF, BTestDF, STrainDF, BTrainDF


def train(train_set, filename="output", dump=True):
	params = read_params()

	print(params)
	train_data = np.array(train_set[:,:11], dtype="float64")
	labels = np.array(train_set[:,14], dtype="float64")

	tree = DecisionTreeClassifier()
	# grad = GradientBoostingClassifier(n_estimators=500, random_state=1, max_depth=3)
	grad = GradientBoostingClassifier(**params)

	t = time.time()
	grad.fit(train_data, labels)
	print(time.time() - t)

	if dump:
		with open(f"{filename}.pickle", "wb") as file:
			pickle.dump(grad, file)


def load(STestDF, BTestDF, filename="output"):
	with open(f"{filename}.pickle", "rb") as file:
		grad = pickle.load(file)

	SResponse = grad.decision_function(STestDF.values[:,:11])
	BResponse = grad.decision_function(BTestDF.values[:,:11])

	STestDF["output"] = SResponse
	BTestDF["output"] = BResponse

	return STestDF, BTestDF


if __name__ == "__main__":
	dir_ind = 0
	for sbd, dirs, files in os.walk("models/"):
		while f"{dir_ind}" in dirs:
			dir_ind += 1
		os.mkdir(f"models/{dir_ind}")
		with open(f"models/{dir_ind}/settings.json", "w") as file:
			json.dump(read_params(), file)
		break

	TrainSubset, STestDF, BTestDF, STrainDF, BTrainDF = dataset_gen()
	train(TrainSubset, filename=f"models/{dir_ind}/model")

	STestDF, BTestDF = load(STestDF, BTestDF, filename=f"models/{dir_ind}/model")
	BDT_output_hist_plot(STestDF, BTestDF, model_id=f"models/{dir_ind}/test_output")
	significance_plot(STestDF, BTestDF, 0.5, model_id=f"models/{dir_ind}/test_sign")

	STrainDF, BTrainDF = load(STrainDF, BTrainDF, filename=f"models/{dir_ind}/model")
	BDT_output_hist_plot(STrainDF, BTrainDF, model_id=f"models/{dir_ind}/train_output")
	significance_plot(STrainDF, BTrainDF, 0.5, model_id=f"models/{dir_ind}/train_sign")
