# %load ../../src/data/data_utils.py
# %%writefile ../../src/data/data_utils.py

"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""


import pandas as pd
import numpy as np
import itertools

def CreatePairwiseRankData(dfDataset):
    
    """Create pairwise ranked data from dataset. All possible combinations are included.
    (see README for default dataset layout)
    
    Parameters
    -----------
    dfDataset : DataFrame
        Dataframe containing at least 'ID', 'sequence', 'mean_score', '35boxstart' and '10boxstart'. 
        'mean_score_sd' is an optional column
    Returns
    --------
    DF : Dataframe 
        Dataframe containing paired data with arguments found in original dataframe (subscripted with '_1' and '_2')
        and rank. Rank is defined as 1 for samples in which 'mean_score_1' > 'mean_score_2' and -1 in other cases
    """
    
    sampleCount = dfDataset.shape[0]
    
    DF = pd.DataFrame(index=range(int(sampleCount*(sampleCount-1)/2)), 
                  columns=[])
    
    ZIP = list(itertools.combinations(dfDataset['ID'],2))

    DF['ID_1'] = [item[0] for item in ZIP]
    DF['ID_2'] = [item[1] for item in ZIP]

    DF['sequence_1'] = [dfDataset[dfDataset['ID']==x]['sequence'].values[0] for x in DF['ID_1']]
    DF['sequence_2'] = [dfDataset[dfDataset['ID']==x]['sequence'].values[0] for x in DF['ID_2']]

    DF['mean_score_1'] = [dfDataset[dfDataset['ID']==x]['mean_score'].values[0] for x in DF['ID_1']]
    DF['mean_score_2'] = [dfDataset[dfDataset['ID']==x]['mean_score'].values[0] for x in DF['ID_2']]
    
    if 'mean_score_sd' in dfDataset.columns:
        DF['mean_score_sd_1'] = [dfDataset[dfDataset['ID']==x]['mean_score_sd'].values[0] for x in DF['ID_1']]
        DF['mean_score_sd_2'] = [dfDataset[dfDataset['ID']==x]['mean_score_sd'].values[0] for x in DF['ID_2']]

    DF['35boxstart_1'] = [dfDataset[dfDataset['ID']==x]['35boxstart'].values[0] for x in DF['ID_1']]
    DF['35boxstart_2'] = [dfDataset[dfDataset['ID']==x]['35boxstart'].values[0] for x in DF['ID_2']]

    DF['10boxstart_1'] = [dfDataset[dfDataset['ID']==x]['10boxstart'].values[0] for x in DF['ID_1']]
    DF['10boxstart_2'] = [dfDataset[dfDataset['ID']==x]['10boxstart'].values[0] for x in DF['ID_2']]

    DF['rank'] = [1 if x>(DF.iloc[i]['mean_score_2']) else -1 for i, x in enumerate(DF['mean_score_1']) ]
    
    return DF
