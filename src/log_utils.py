# %load ../../src/log_utils.py
# %%writefile ../../src/log_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import json
import datetime as dt
def LogInit(function, model, parameters, localarg):
    time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M')            
    parString = ''.join([num for num in parameters])
    LOGFILENAME = '{}_{}_{}-{}-{}_{}'.format(time ,function, model[0].upper(),
                                            model[1],model[2],parString)
    RESULTLOG = '../logs/result_logger/'+LOGFILENAME
    
    MAINLOG = '../logs/log.txt'
    output = '\n\nSTARTED  '+LOGFILENAME + '\n\targuments: '+localarg
    with open(MAINLOG, 'a') as f:
        f.write(output)
    f.close()
    print(output)
    return MAINLOG, RESULTLOG
    
def LogWrap(MAINLOG, RESULTLOG, results):
    
    output=''
    if type(results) is list:
        for result in results:
            output=output+'\n'+result
    else:
        output = results
    with open(RESULTLOG+'.txt', 'w') as f:
        f.write(output)
    f.close()
    outputWrap = '\n...FINISHED'
    with open(MAINLOG, 'a') as f:
        f.write(outputWrap)
    print(outputWrap)
    f.close()
    
                