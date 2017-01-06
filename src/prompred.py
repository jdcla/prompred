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




def runGS(data, modelCore, modelOpt, parLabel, parRange, seqRegions, k=5, kInner= 4, pw=False, shuffle=True):
	
	dfDataset, featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=shuffle)
	X = featureBox.values
	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']])

	parModel = mu.SetupModel(modelCore, modelOpt)
	parSet = mu.GetParameterSet(parLabel, parRange)

	model = mu.SelectModel(**parModel)
	GS = GridSearchCV(model, parSet, cv=k)
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

	return results, figures


def runLCis(data, modelCore, modelOpt, seqRegions, k=5, kInner=4, pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)
	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']],dtype=np.dtype(float))
	parModel = mu.SetupModel(modelCore, modelOpt)
	scores = mu.LearningCurveInSample(dfDataset, featureBox, y ,parModel, k=k, pw=pw, step=step)

	fig, ax1, ax2 = pu.LearningCurveInSample(scores, k, step)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures

def runLCisE(data, dataEnrich, modelCore, modelOpt, seqRegions, k=5, kInner=4, pw=False, shuffle=True, step=1):
	
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)
	dfEnrich, enrichBox = fu.CreateFeaturesFromData(dataEnrich, seqRegions, pw, shuffle=True)

	X_enrich = enrichBox.values
	y_enrich = dfEnrich['rank']

	X = featureBox.values
	y = dfDataset['rank']

	parModel = mu.SetupModel(modelCore, modelOpt)

	scores = mu.LearningCurveInSampleEnriched(dfDataset, featureBox, enrichBox, y, y_enrich, parModel, k, pw, step)

	fig, ax1, ax2 = pu.LearningCurveInSampleEnriched(scores, k, step)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures

def runLCos(data, dataOut, dataOutLabels, modelCore, modelOpt, seqRegions, pw=False, shuffle=True, step=1):
	
	parModel = mu.SetupModel(modelCore,modelOpt)

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

	X = featureBox.values
	if pw is True:
		y = dfDataset['rank']
	else:
		y = np.array([math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']])
		
	scores = mu.LearningCurveOutOfSample(dfDataset, featureBox, y , dataList, parModel, pw=pw, step=step)

	fig, ax = pu.LearningCurveOutOfSample(scores, k, dataOutLabels)
	
	results = str(scores)
	figures = [fig]
	
	return results, figures
	
def runLCosE(data, dataEnrich, dataOut, dataOutLabels, modelCore, modelOpt, seqRegions, pw=False, shuffle=True, step=1):
	
	parModel = mu.SetupModel(modelCore,modelOpt)
	dfDataset , featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=True)

	X = featureBox.values
	y = dfDataset['rank']

	enrichBoxList = []
	y_enrich = []

	for e in dataEnrich:
		dfEnrich, enrichBox = fu.CreateFeaturesFromData(e, seqRegions, pw)
		y_en = dfEnrich['rank']
		enrichBoxList.append(enrichBox)
		y_enrich.append(y_en)
		
	enrichBox = np.vstack((enrichBoxList[:]))

	dataOutList = []

	for d in dataOut:
		dfOut, outBox = fu.CreateFeaturesFromData(d, seqRegions, pw)
		y_out = dfOut['rank']
		dataOutList.append((dfOut, outBox, y_out))


	scores = mu.LearningCurveOutOfSampleEnriched(dfDataset, featureBox, enrichBox, y, y_enrich, dataOutList, parModel, pw, step)
	fig, ax = pu.LearningCurveOutOfSampleEnriched(scores, step, dataOutLabels)
	
	results = str(scores)
	figures = [fig]
	
def ExecuteFunction(function, data, dataEnrich, dataOut, dataOutLabel, seqRegions, modelCore, modelOpt, parLabel, parRange,k,kInner, pw, shuffle, step):
	localarg = str(locals())
	MAINLOG, RESULTLOG = log.LogInit(function, modelCore, parLabel, localarg)

	if function == 'GS':
		results, figures = runGS(data, modelCore, modelOpt, parLabel, parRange, seqRegions, k, kInner, pw, shuffle)
	elif function == 'LCis':
		results, figures = runLCis(data, modelCore, modelOpt, seqRegions, k, kInner, pw, shuffle, step)
	elif function == 'LCisE':
		results, figures = runLCisE(data, dataEnrich, modelCore, modelOpt, seqRegions, k, kInner, pw, shuffle, step)
	elif function == 'LCos':
		results, figures = runLCos(data, dataOut, dataOutLabel, modelCore, modelOpt, seqRegions, pw, shuffle, step)
	index=0
	for fig in figures:
		fig.savefig(RESULTLOG+str(index)+'.png')
		index+=1
		
	log.LogWrap(MAINLOG, RESULTLOG, results)

def main():
	parser = argparse.ArgumentParser(description='high-end script function for prompred')
	parser.add_argument('function', type=str, choices=('GS', 'LCis', 'LCisE', 'LCos', 'LCosE'), help="function to execute")
	parser.add_argument('-d', '--dataset', type=str, help='path to dataset, see dataset description for requirements')
	parser.add_argument('-de', '--data_enrich', type=str, dest='dataEnrich', help='path to dataset, used in enriched learning curves')
	parser.add_argument('-do', '--data_out', type=str, dest='dataOut', nargs='*', help='path to dataset, used for out of sample learning curves')
	parser.add_argument('-dol','--data_out_label', type=str, dest='dataOutLabel', nargs='*', help='labels of the datasets given for --data_out')
	parser.add_argument('-s', '--seqReg', type=int, nargs=4, help='positional arguments of nucleotide sequences respective to the start of the 35-box and 10-box.')
	parser.add_argument('-m', '--model', type=str, choices=('ridge',  'SVC', 'SVR', 'OLS', 'lasso', 'huber', 'treeReg', 'treeClass', 'forestReg', 'forestClass'), help="")
	parser.add_argument('-k', '--kernel', type=str, choices=('poly','rbf','sigmoid'), default=None, help="")
	parser.add_argument('-g', '--grade', type=int, default=None, help="")
	parser.add_argument('-o', '--modelOpt', type=json.loads, default='{}', help="")
	parser.add_argument('--parL', type=str, default="", choices=("alpha","gamma","C","coef0","epsilon","max_depth","min_samples_split","max_features"), dest="parLabel", nargs='*', help="nucleotide regions of interest for their respective reference regions")
	parser.add_argument('--parR', type=int, dest="parRange", nargs='*', help="nucleotide regions of interest for their respective reference regions")
	parser.add_argument('-K', type= int, default=5, dest='k', help="amount of folds used for cross-validation")
	parser.add_argument('-KN', type=int, default=4, dest='kInner', help="amount of folds used for inner cross-validation")
	parser.add_argument('-step', type=int, default=1, help="amount of promoters added to the training data for each step")
	parser.add_argument('-pw', '--pairwise', action="store_true", help="use of pairwise data")
	parser.add_argument('-ns', '--noshuffle', action="store_true", help="disables shuffling of data")
	
	
	args = parser.parse_args()
	seqRegions=np.array([[args.seqReg[0],args.seqReg[1]],[args.seqReg[2],args.seqReg[3]]])
	modelCore = [args.model,args.kernel,args.grade]
	shuffle= args.noshuffle==False
	#modelOpt = json.loads(args.modelOpt)
	ExecuteFunction(args.function, args.dataset, args.dataEnrich, args.dataOut, args.dataOutLabel, seqRegions, modelCore, args.modelOpt, args.parLabel, args.parRange, args.k, args.kInner, args.pairwise, shuffle, args.step)



if __name__ == "__main__":
    sys.exit(main())


