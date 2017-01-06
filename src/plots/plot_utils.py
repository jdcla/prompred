# %load ../../src/plots/plot_utils.py
# %%writefile ../../src/plots/plot_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def HeatmapCVSinglePar(GS):
   
   if len(GS.param_grid) > 1:
       print("Does not accept more than 1 parameter")

   k = GS.n_splits_

   scores = np.array([])
   for fold in range(k):
       scores = np.append(scores,GS.cv_results_['split{}_test_score'.format(fold)])

   scores = scores.reshape(k,-1)

   for key in GS.param_grid:

       extent=[GS.param_grid[key].min(),GS.param_grid[key].max(),k+0.5,0.5]

       fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
       fig.suptitle("{}".format(key),fontsize=14, 
                fontweight='bold')
       ax1.set_title("Cross Validation") 
       ax1.set_yticks(np.arange(k+1))
       cax1 = ax1.imshow(scores, cmap='hot', vmin=0 , 
                         interpolation='nearest',extent=extent)
       fig.colorbar(cax1, ax=ax1)

       ax2.set_title("Cross Validation: Normalized")
       cax2 = ax2.imshow(sklearn.preprocessing.normalize(scores, axis=1), cmap='hot', vmin=0, 
                         interpolation='nearest',extent=extent)
       fig.colorbar(cax2, ax=ax2)

       ax1.set_xlabel('parameter value'), ax2.set_xlabel('parameter value')
       ax1.set_ylabel('k fold'), ax2.set_ylabel('k fold')
       ax1.set_xscale('log'), ax2.set_xscale('log')
       
       return fig, (ax1, ax2)
       
def HeatmapCVDoublePar(GS):
   
   if len(GS.param_grid) is not 2:
       print("Model has to evaluate two parameters")

   keyLengths = []
   keys = []
   for key in GS.param_grid:
       keys.append(key)
       keyLengths.append(len(GS.param_grid[key]))

   scores = np.flipud(GS.cv_results_["mean_test_score"].reshape(keyLengths))

   xMin, xMax = GS.param_grid[keys[0]].min(), GS.param_grid[keys[0]].max()
   yMin, yMax = GS.param_grid[keys[1]].min(), GS.param_grid[keys[1]].max()

   fig, ax = plt.subplots()
   ax.set_title("Model accuracy wrt. model parameters")
   cax = ax.imshow(scores, cmap='hot', interpolation='none', 
                   vmin=0, vmax=np.max(GS.cv_results_["mean_test_score"]),  
                   extent=[xMin,xMax,yMin,yMax], 
                   aspect="auto", 
             )
   fig.colorbar(cax)
   ax.set_xlabel(keys[0], fontweight='bold'), plt.xscale('log')
   ax.set_ylabel(keys[1], fontweight='bold'), plt.yscale('log')

   return fig, ax

def ScatterPvT(model, X, yTrue):
   
   yPred = model.predict(X)
   yMax = max(yTrue) if max(yTrue)>max(yPred) else max(yPred)
   
   fig, ax = plt.subplots( figsize=(6,4))
   ax.set_title("pred/true: K-fold")
   ax.plot([0,yMax],[0,yMax], ls="--", c=".3")
   ax.plot(yTrue,yPred, 'bo')
   ax.set_xlabel("True label")
   ax.set_ylabel("Predicted label")
   ax.text(0.05,0.9, 'score: '+ str(model.score(X,yTrue)),
       fontsize=10, transform=ax.transAxes)

   return fig, ax

def LearningCurveInSample(scores, k, step):
   
   fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,8),sharex=True)

   ax1.set_title("Learning curve in sample score, stepwise increase of promoter in train library")
   for i in range(k):
       colors = ['bo','ro','yo','go','wo','mo','co','ko','bo','co']
       ax1.plot(range(5,len(scores[i,:])+5),scores[i,:], colors[i])
   ax1.set_xlabel("Step")
   ax1.set_ylabel("Score")
   meanScores=np.mean(scores,axis=0)
   stdScores=np.std(scores,axis=0)
   ax2.errorbar(range(5,len(meanScores)+5), meanScores[:], stdScores[:])
   ax2.set_xlabel("Step")
   ax2.set_ylabel("Score")
   ax2.set_xticklabels(range(5,len(scores[i,:])*step+5,step))
   
   return fig, ax1, ax2

def LearningCurveInSampleEnriched(scores, k, step):
   
   fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,8),sharex=True)

   ax1.set_title("Learning curve in sample score of enriched dataset")
   for i in range(k):
       colors = ['bo','ro','yo','go','wo','mo','co','ko','bo','co']
       ax1.plot(range(len(scores[i,:])),scores[i,:], colors[i])
   ax1.set_xlabel("Step")
   ax1.set_ylabel("Score")
   meanScores=np.mean(scores,axis=0)
   stdScores=np.std(scores,axis=0)
   ax2.errorbar(range(len(meanScores)), meanScores[:], stdScores[:])
   ax2.set_xlabel("Step")
   ax2.set_ylabel("Score")
   
   return fig, ax1, ax2

def LearningCurveOutOfSample(scores, step, labels):
   fig, ax = plt.subplots(1,1, figsize=(8,6))

   ax.set_title("Learning curve out of sample score, stepwise increase of promoter in train library")
   colors = ['bo','ro','yo','go','wo','mo','co','ko','bo','co']
   for j in range(len(scores)):
       ax.plot(range(5, len(scores[j,:])+5),scores[j,:], colors[j], label=dataOutLabels[j])
   ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   ax.set_xlabel("Step")
   ax.set_ylabel("Score")
   ax.set_xticklabels(np.int64((range(len(scores))*step)+5))

   return fig, ax

def LearningCurveOutOfSampleEnriched(scores, step, labels):
   fig, ax = plt.subplots(1,1, figsize=(8,6))

   ax.set_title("Learning curve out of sample score, stepwise increase of promoter in train library")
   colors = ['bo','ro','yo','go','wo','mo','co','ko','bo','co']
   for j in range(len(scores)):
       ax.plot(range(len(scores[j,:])),scores[j,:], colors[j], label=labels[j])
   ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   ax.set_xlabel("Step")
   ax.set_ylabel("Score")
   ax.set_xticklabels(np.int64((np.arange(len(scores))*step)))

   return fig, ax