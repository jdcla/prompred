import sys
import json
import argparse
from argparse import RawTextHelpFormatter
import features.feature_utils as fu
import models.model_utils as mu
import plots.plot_utils as pu
import log_utils as log
import math
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd


def runGS(data, modelCore, modelOpt, parLabel, parRange, seqRegions, k=5, kInner= 4, pw=False, shuffle=True, n_jobs=4):
	
	
	dfDataset, featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=shuffle)
	X = featureBox.values
	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']])

	parModel = mu.SetupModel(modelCore, modelOpt)
	parSet = mu.GetParameterSet(parLabel, parRange)

	model = mu.SelectModel(**parModel)
	GS = GridSearchCV(model, parSet, cv=k, n_jobs=n_jobs, verbose=2)
	GS.fit(X,y)
	
	fig1, ax = pu.ScatterPvT(GS, X, y)
	figures = [fig1]
	if len(parLabel) is 2:
		fig2, ax = pu.HeatmapCVDoublePar(GS)
		figures.append(fig2)
	if len(parLabel) is 1:
		fig2, ax = pu.HeatmapCVDoublePar(GS)
		figures.append(fig2)

	results = [str(GS.best_estimator_.get_params()),str(GS.best_score_),str(GS.cv_results_)]

	return model, results, figures


def runLCis(data, dataLabel, modelCore, modelOpt, seqRegions, scoring='default', k=5, kInner=4, pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)
	
	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']],dtype=np.dtype(float))
	
	parModel = mu.SetupModel(modelCore, modelOpt)
	
	scores = mu.LearningCurveInSample(dfDataset, featureBox, y ,parModel, scoring, k=k, pw=pw, step=step)

	fig, ax1, ax2 = pu.LearningCurveInSample(scores, k, step, dataLabel, scoring)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures

def runLCisE(data, dataLabel, dataEnrich, dataEnrichLabel, modelCore, modelOpt, seqRegions, scoring='default', k=5, kInner=4, pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=shuffle)
	dfEnrich, enrichBox = fu.CreateFeaturesFromData(dataEnrich, seqRegions, pw, shuffle=shuffle)

	y = dfDataset['rank']
	y_enrich = dfEnrich['rank']
	
	parModel = mu.SetupModel(modelCore, modelOpt)

	scores = mu.LearningCurveInSampleEnriched(dfDataset, featureBox, enrichBox, y, y_enrich, parModel, scoring, k, pw, step)

	fig, ax1, ax2 = pu.LearningCurveInSampleEnriched(scores, k, step, dataLabel, dataEnrichLabel, scoring)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures

def runLCos(data, dataLabel, dataOut, dataOutLabel, modelCore, modelOpt, seqRegions, scoring='default', pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)
	dataList = []
	for d in dataOut:
		
		if pw is True:
			dfShuffled, Box = fu.CreateFeaturesFromData(d, seqRegions, pw)
			y_out = dfShuffled['rank']
		else:
			dfShuffled, Box = fu.CreateFeaturesFromData(d, seqRegions, pw)
			y_out = dfShuffled['mean_score']
		dataList.append((dfShuffled, Box, y_out))

	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']])
		
	parModel = mu.SetupModel(modelCore,modelOpt)

	scores = mu.LearningCurveOutOfSample(dfDataset, featureBox, y , dataList, parModel, scoring=scoring, pw=pw, step=step)

	fig, ax = pu.LearningCurveOutOfSample(scores, step, dataOutLabel, dataLabel, scoring)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures
	
def runLCosE(data, dataLabel, dataEnrich, dataEnrichLabel, dataOut, dataOutLabel, modelCore, modelOpt, seqRegions, scoring='default', pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)
	dfEnrich, enrichBox = fu.CreateFeaturesFromData(e, seqRegions, pw)
	
	dataOutList = []
	for d in dataOut:
		dfOut, outBox = fu.CreateFeaturesFromData(d, seqRegions, pw)
		y_out = dfOut['rank']
		dataOutList.append((dfOut, outBox, y_out))
	
	y = dfDataset['rank']
	y_enrich = dfEnrich['rank']
	
	scores = mu.LearningCurveOutOfSampleEnriched(dfDataset, featureBox, enrichBox, y, y_enrich, dataOutList, parModel, scoring, pw, step)
	
	fig, ax = pu.LearningCurveOutOfSampleEnriched(scores, step, dataOutLabel, dataLabel, dataEnrichLabel, scoring)
	
	results = str(scores)
	figures = [fig]

	return results, figures
	
def ExecuteFunction(function, data, dataLabel, dataEnrich, dataEnrichLabel, dataOut, dataOutLabel, seqRegions, modelCore, modelOpt, parLabel, parRange, scoring, k,kInner, pw, shuffle, step, n_jobs):
	localarg = locals()
	LOGFILENAME, MAINLOG, RESULTLOG = log.LogInit(function, modelCore, parLabel, localarg)

	if function == 'GS':
		model, results, figures = runGS(data, modelCore, modelOpt, parLabel, parRange, seqRegions, k, kInner, pw, shuffle, n_jobs)
	elif function == 'LCis':
		results, figures = runLCis(data, dataLabel, modelCore, modelOpt, seqRegions, scoring, k, kInner, pw, shuffle, step)
	elif function == 'LCisE':
		results, figures = runLCisE(data, dataLabel, dataEnrich, dataEnrichLabel, modelCore, modelOpt, seqRegions, scoring, k, kInner, pw, shuffle, step)
	elif function == 'LCos':
		results, figures = runLCos(data, dataLabel, dataOut, dataOutLabel, modelCore, modelOpt, seqRegions, scoring, pw, shuffle, step)
	elif function == 'LCosE':
		results, figures = runLCosE(data, dataLabel, dataEnrich, dataEnrichLabel, dataOut, dataOutLabel, modelCore, modelOpt, seqRegions, scoring, pw, shuffle, step)
	
	if model is not None:
		joblib.dump(model, "../models/" +LOGFILENAME+"_model.sav")
	
	index=0
	for fig in figures:
		if fig.gca().legend_ is None:
			print(RESULTLOG)
			print(index)
			fig.savefig(RESULTLOG+str(index)+'.png')
		else:
			lgd = fig.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
			fig.savefig(RESULTLOG+str(index)+'.png', bbox_extra_artists=(lgd,),bbox_inches='tight')
		index+=1
		
	log.LogWrap(MAINLOG, RESULTLOG, results)

def main():
	parser = argparse.ArgumentParser(description='high-end script function for prompred')
	parser.add_argument('function', type=str, choices=('GS', 'LCis', 'LCisE', 'LCos', 'LCosE'), help="function to execute")
	parser.add_argument('-d', '--data', type=str, nargs='*', help='path to dataset, see dataset description for requirements')
	parser.add_argument('-dl','--data_label', type=str, dest='dataLabel', help='label of the dataset given for --data')
	parser.add_argument('-de', '--data_enrich', type=str, dest='dataEnrich', nargs='*', help='path to dataset, used in enriched learning curves')
	parser.add_argument('-del','--data_enrich_label', type=str, dest='dataEnrichLabel', nargs='*', help='labels of the datasets given for --data_enrich')
	parser.add_argument('-do', '--data_out', type=str, dest='dataOut', nargs='*', help='path to dataset, used for out of sample learning curves')
	parser.add_argument('-dol','--data_out_label', type=str, dest='dataOutLabel', nargs='*', help='labels of the datasets given for --data_out')
	parser.add_argument('-s', '--seqReg', type=int, nargs=4, help='positional arguments of nucleotide sequences respective to the start of the 35-box and 10-box.')
	parser.add_argument('-m', '--model', type=str, choices=('ridge',  'SVC', 'SVR', 'OLS', 'lasso', 'huber', 'treeReg', 'treeClass', 'forestReg', 'forestClass'), help="")
	parser.add_argument('-k', '--kernel', type=str, choices=('poly','rbf','sigmoid'), default=None, help="The kernel function. Used with 'ridge', 'SVC', or 'SVR")
	parser.add_argument('-g', '--grade', type=int, default=None, help="grade of the function. Used with any kernel or the OLS, lasso, huber or ridge model")
	parser.add_argument('-o', '--modelOpt', type=json.loads, default='{}', help="Optional hyperparameters given to the model. Available parameters, with defaults: alpha=0.1, gamma=0.1, epsilon=0.1, coef0=1, "
						"fitInt=True, normalize=True, max_depth=None, max_features=None, min_samples_split = 2, n_estimators = 50, C=1, n_jobs=12")
	parser.add_argument('--parL', type=str, default="", choices=("alpha","gamma","C","coef0","epsilon","max_depth","min_samples_split","max_features"), dest="parLabel", nargs='*', help="nucleotide regions of interest for their respective reference regions")
	parser.add_argument('--parR', type=int, dest="parRange", nargs='*', help="nucleotide regions of interest for their respective reference regions")
	parser.add_argument('-sc', '--scoring', type=str, default='kt', help="scoring method used")
	parser.add_argument('-K', type= int, default=5, dest='k', help="amount of folds used for cross-validation")
	parser.add_argument('-KN', type=int, default=4, dest='kInner', help="amount of folds used for inner cross-validation")
	parser.add_argument('-step', type=int, default=1, help="amount of promoters added to the training data for each step")
	parser.add_argument('-pw', '--pairwise', action="store_true", help="use of pairwise data")
	parser.add_argument('-ns', '--noshuffle', action="store_true", help="disables shuffling of data")
	parser.add_argument('-njobs', type=int, default=4, help="amount of jobs to run in parallel, only supported by some functions")
	
	# choices("default","kt", "spearman"),
	args = parser.parse_args()
	seqRegions=np.array([[args.seqReg[0],args.seqReg[1]],[args.seqReg[2],args.seqReg[3]]])
	modelCore = [args.model,args.kernel,args.grade]
	shuffle= args.noshuffle==False
	ExecuteFunction(args.function, args.data, args.dataLabel, args.dataEnrich, args.dataEnrichLabel, args.dataOut, args.dataOutLabel, seqRegions, 
					modelCore, args.modelOpt, args.parLabel, args.parRange, args.scoring, args.k, args.kInner, args.pairwise, shuffle, args.step, args.njobs)



if __name__ == "__main__":
    sys.exit(main())


