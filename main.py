import numpy as np
import ROOT as root
import pandas as pd
import pickle

from array import array

from config import *
from dataloader import dataset_gen
from helpers import setup
from plotters import BDT_output_hist_plot, significance_plot, clear_file

from sys import argv


def build_reader(DF, uploadfile="output"):
	with open(f"models/{uploadfile}.pickle", "rb") as file:
		grad = pickle.load(file)

	SDF, BDF = DF[DF["classID"] == 1], DF[DF["classID"] == 0]

	SResponse = grad.decision_function(SDF.values[:,:11])
	BResponse = grad.decision_function(BDF.values[:,:11])

	SDF["BDToutput"] = SResponse
	BDF["BDToutput"] = BResponse

	return SDF, BDF


def main():
	setup()
	uploadfile = argv[1]
	clear_file(uploadfile)

	TrainDF, TestDF = dataset_gen(backgrounds="train")
	TrainSigDF, TestSigDF = dataset_gen(backgrounds="all", region="signal")
	SigDF = pd.concat((TrainSigDF, TestSigDF), ignore_index=True)

	STrainDataframe, BTrainDataframe = build_reader(TrainDF, uploadfile)
	STestDataframe, BTestDataframe 	 = build_reader(TestDF, uploadfile)

	SDataframe, BDataframe 			 = build_reader(SigDF, uploadfile)

	BDT_output_hist_plot(STestDataframe, BTestDataframe,
						 STrainDataframe, BTrainDataframe, model_id=uploadfile)
	# significance_plot(SDataframe, BDataframe, ratio=1, model_id=uploadfile)


if __name__ == "__main__":
	main()

	







