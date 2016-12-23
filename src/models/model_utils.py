# %load ../../src/models/model_utils.py
# %%writefile ../../src/models/model_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""


import pandas as pd
import numpy as np
import sklearn as sk


def GetParameterValues(parLabel, parRange):
    
    """Retrieve a set of parameter values used for training of a model in sklearn. 
    
    Parameters
    -----------
    parLabel : 1-dimensional numpy array (str)
        numpy array holding a set of parameter labels. Valid labels include:
        [alpha, gamma, C, coef0, epsilon, max_depth, min_samples, max_features]
        
    parRange : 1-dimensional numpy array (int)
        numpy array with the amount of parameters returned for every parameter label. 
        parLabel and parRange must be of the same dimension.
        
    
    Returns
    --------
    parEval : Dictionary 
        Dictionary containing a set of parameters for every label
    """
   
    
    parameters = [np.zeros(parRange[u]) for u in range(len(parRange))]
 
    for i in range(len(parLabel)):
        if parLabel[i] is "alpha":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "gamma":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "C":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "coef0":
            parameters[i][:] = [math.pow(10,(u - np.around(parRange[i]/2))) for u in range(parRange[i])]
        elif parLabel[i] is "epsilon":
            parameters[i][:] = [0+2/parRange[i]*u for u in range(parRange[i])]
        elif parLabel[i] is "max_depth":
            parameters[i][:] = [u+1 for u in range(parRange[i])]
        elif parLabel[i] is 'min_samples':
            parameters[i][:] = [u+1 for u in range(parRange[i])]
        elif parLabel[i] is 'max_features':
            parameters[i][:] = [int(u+2) for u in range(parRange[i])]
        else:
            return print("not a valid parameter")
    
    parEval = {parLabel[u]:parameters[u] for u in range(len(parLabel))}
    
    return parEval