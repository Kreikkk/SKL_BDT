import numpy as np


def chisq(hist_true, hist_exp, tp):
	chisq = hist_true.Chi2Test(hist_exp, option="WW CHI2")
	p_val = hist_true.Chi2Test(hist_exp, option="WW")
	chisq_over_ndof = hist_true.Chi2Test(hist_exp, option="WW CHI2/NDF")

	return chisq, chisq_over_ndof, p_val