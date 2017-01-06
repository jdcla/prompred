from sklearn.model_selection import GridSearchCV




def runGS(data, modelCore, modelOpt, parLabel, parRange, seqRegions, k=5, kInner= 4, pw=False, shuffle=True):
	
	dfDataset, featureBox = fu.CreateFeaturesFromData(data, seqRegions, pw, shuffle=shuffle)
	X = featureBox.values
	y = [math.sqrt(math.sqrt(u)) for u in dfDataset['mean_score']]

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
