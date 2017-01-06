# %load ../../src/feature/feature_utils.py
# %%writefile ../../src/features/feature_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""


import pandas as pd
import numpy as np


def AlignSequences(dfSequences, pw=False):
    
    """Align sequences with eachother in the dataframe, adding '-' for 
    missing nucleotides before and after shorter sequences, and changing the 35- and 10-box 
    reference regions accordingly
    
    Parameters
    -----------
    dfSequences : DataFrame
        Dataframe containing columns sequence and reference start regions
        DataFrame.columns has to contain ['sequence', '35boxstart', '10boxstart']
    
    Returns
    --------
    dfAlignedSequences : Dataframe 
        Dataframe containing aligned sequences
    """
    if pw is True:
        
        start35Box_1 = dfSequences['35boxstart_1'].values 
        start35Box_2 = dfSequences['35boxstart_2'].values 
        start35BoxMax = start35Box_1.max() if start35Box_1.max()>start35Box_2.max() else start35Box_2.max()
        
        dfAlignedSequences = dfSequences.copy()
          
        for i in range(1,3):
            
            sequences = dfSequences['sequence_{}'.format(i)].values 
            start35Box = dfSequences['35boxstart_{}'.format(i)].values 
            start10Box = dfSequences['10boxstart_{}'.format(i)].values 

            difLength = start35BoxMax-start35Box

            sequenceAligned = ["-" *difLength[u]+sequences[u] if difLength[u]>=0 else sequences[u][abs(difLength[u]):] for u in range(np.shape(sequences)[0]) ]
            start35Box = np.array([start35Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])
            start10Box = np.array([start10Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])

            maxLength = max([len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])])
            difLength = [maxLength - len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])]
            sequences = [sequenceAligned[u]+ '-'*difLength[u] for u in range(np.shape(sequenceAligned)[0]) ]

            dfAlignedSequences['sequence_{}'.format(i)] = sequences
            dfAlignedSequences['35boxstart_{}'.format(i)] = start35Box
            dfAlignedSequences['10boxstart_{}'.format(i)] = start10Box


        
    else:
        
        sequences = dfSequences['sequence'].values 
        start35Box = dfSequences['35boxstart'].values 
        start10Box = dfSequences['10boxstart'].values 

        difLength = start35Box.max()-start35Box

        sequenceAligned = ["-" *difLength[u]+sequences[u] if difLength[u]>=0 else sequences[u][abs(difLength[u]):] for u in range(np.shape(sequences)[0]) ]
        start35Box = np.array([start35Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])
        start10Box = np.array([start10Box[u]+difLength[u] for u in range(np.shape(sequences)[0])])

        maxLength = max([len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])])
        difLength = [maxLength - len(sequenceAligned[u]) for u in range(np.shape(sequenceAligned)[0])]
        sequences = [sequenceAligned[u]+ '-'*difLength[u] for u in range(np.shape(sequenceAligned)[0]) ]

        dfAlignedSequences = dfSequences.copy()
        dfAlignedSequences['sequence'] = sequences
        dfAlignedSequences['35boxstart'] = start35Box
        dfAlignedSequences['10boxstart'] = start10Box

    return dfAlignedSequences


def CreateDummyNucleotideFeatures(sequences, posRange):
    
    """Create dummy dataframe of nucleotides for two regions surrounding 35- and 10-box
    
    Parameters
    -----------
    sequences : 1-dimensional numpy array
        numpy array containing an array of sequences (str)
        
    posRange : tuple, 2-element array
        tuple containing range of the sequence off of which dummy features will be created. This range is 
        used to create column names for the obtained dummy features
          
    Returns
    --------
    dfDummyDataFrame : Dataframe 
        Dataframe containing dummy features
    """
    
    # Create Position Matrix
    nucleotideMatrix = ChopStringVector(sequences)
    # Convert to Dataframe
    posRangeCol = [str(x) for x in range(posRange[0],posRange[1])]
    dfNucleotideMatrix = pd.DataFrame(nucleotideMatrix, columns = posRangeCol)

    # Create dummy Matrix
    dfDummyNucleotideFeatures =  pd.get_dummies(dfNucleotideMatrix)


    return dfDummyNucleotideFeatures


def ChopStringVector(strings):
    
    """Chops a vector of strings (equal length) into a matrix of characters, with each row containing a separate 
    string
    
    Parameters
    -----------
    strings : 1-dimensional numpy array
        numpy array containing array of strings 
   
    Returns
    --------
    charMatrix : 2-dimensional numpy array 
        Matrix containing chopped up strings
    """
    
    charMatrix = np.empty([len(strings),len(strings[0])],dtype=np.dtype(str,1))
    strings=np.array(strings)
    for i in range(0,len(strings)):
        charMatrix[i] = [strings[i][u] for u in range(0,len(strings[i]))]

    return charMatrix 

def CreateFeaturesFromData(data, seqRegions, pw, shuffle=True):
    
    """Creates features from
    
    data: string
        PATH or filename of dataset
    
    seqRegions : tuple,list
        List containing two positional ranges from which features are derived, respectively starting from first
        nucleotide of 35-box and 10-box
        Example: [[0,6],[0,6]] returns positional features of the range of the -35box and -10box respectively
   
    
    
    """
     
    dfDataset = pd.read_csv(data)
    dfDatasetAligned = AlignSequences(dfDataset, pw)
   
    if shuffle is True:        
        if pw is True:
            dfDatasetShuffled , featureBox = PositionalFeaturesPW(dfDatasetAligned, seqRegions, shuffle=shuffle)
        else:
            dfDatasetShuffled , featureBox = PositionalFeatures(dfDatasetAligned, seqRegions, shuffle=shuffle)
            
        return dfDatasetShuffled , featureBox
    
    if shuffle is False:
        if pw is True:
            featureBox = PositionalFeaturesPW(dfDatasetAligned, seqRegions, shuffle=shuffle)
        else:
            featureBox = PositionalFeatures(dfDatasetAligned, seqRegions, shuffle=shuffle)
        
        return dfDatasetShuffled , featureBox


def CreateFullDummyDataFrame(posRange):
    
    """Creates a dummy nucleotide feature dataframe over a specified range for promotors. '-' is added for 
    nucleotide positions <-35 or >-3
    
    Parameters
    -----------
    posRange : tuple
        Range over which the full nucleotide dummy dataframe is created 
     
    Returns
    --------
    fullDummyDataframe : DataFrame
        Dataframe containing all possible dummy features for positional nucleotides 
    """    
    
    posRangeCol = [str(x) for x in range(posRange[0],posRange[1])]
    
    length = len(posRangeCol)
    a = np.empty([length],dtype=np.dtype(str,1))
    c = np.empty([length],dtype=np.dtype(str,1))
    t = np.empty([length],dtype=np.dtype(str,1))
    g = np.empty([length],dtype=np.dtype(str,1))
    dash = np.empty([length],dtype=np.dtype(str,1))
    a.fill('A')
    t.fill('T')
    c.fill('C')
    g.fill('G')
    dash.fill('A')
    dash[:(-posRange[0]-35)]='-'
    dash[(-posRange[0]-3):]='-'
    dataframe = pd.DataFrame(np.vstack((a,t,c,g,dash)),columns=posRangeCol)
    fullDummyDataFrame = pd.get_dummies(dataframe)
    
    return fullDummyDataFrame

def PositionalFeatures(dfDataset, seqRegions, shuffle=False):
    
    """Create position features for a given promoter dataset. Returns dummy dataset.
    
    Parameters
    -----------
    dfDataset : DataFrame
        Dataframe containing columns sequence and reference start regions
        columnNames = (sequence, 35boxstart, 10boxstart)
        
    seqRegions : tuple,list
        List containing two positional ranges from which features are derived, respectively starting from first
        nucleotide of 35-box and 10-box
        Example: [[0,6],[0,6]] returns positional features of the range of the -35box and -10box respectively.
        
    shuffle : Boolean
        Shuffles input dataset
        
    Returns
    --------
        
    dfDataset : DataFrame
        Shuffled dataset
        
    dfPositionBox : DataFrame 
        Dataframe containing dummy arguments

    
    """
    
    if shuffle is True:
        dfDataset = dfDataset.reindex(np.random.permutation(dfDataset.index))

    # Selecting regions
    
    dfDataset['sequence'] = dfDataset['sequence'].str.upper()
    sequences = dfDataset['sequence'].values 
    start35Box = dfDataset['35boxstart'].values 
    start10Box = dfDataset['10boxstart'].values 
    
    seqRegions = np.array(seqRegions)
    posRange = [seqRegions[0,0]-35,seqRegions[1,1]-12]
        
    reg35 = np.array(seqRegions[0])
    reg10 = np.array(seqRegions[1])
    
    box35 = SelectRegions(sequences, reg35, start35Box)
    box10 = SelectRegions(sequences, reg10, start10Box)
    spacer = start10Box-start35Box-6
    
    spacerM = [u-17 if u-17>0 else 0 for u in spacer]
    spacerL = [abs(u-17) if u-17<0 else 0 for u in spacer]
    
    spacerBox = pd.DataFrame({'spacer_more':spacerM,'spacer_less':spacerL})
   
    # Creating features

    positionBox35 = CreateDummyNucleotideFeatures(box35, reg35-35)
    positionBox10 = CreateDummyNucleotideFeatures(box10, reg10-12)

    positionBoxSS = pd.concat([positionBox35,positionBox10], axis=1)
    
    dfTemplateDummyBox = CreateFullDummyDataFrame(posRange)
    dfFinalBox = pd.DataFrame(0, range(len(sequences)),columns=dfTemplateDummyBox.columns)
    
    colNamesResult = positionBoxSS.columns
    for i in colNamesResult:
        if i in dfFinalBox.columns:
            dfFinalBox[i] = positionBoxSS[i]
    
    dfPositionBox = pd.concat([dfFinalBox, spacerBox], axis=1)

    
    if shuffle is True:
        return dfDataset, dfPositionBox
    else:
        return dfPositionBox

def PositionalFeaturesPW(dfDataset, seqRegions, shuffle=False, subtract=True):
    
    """Create position features of a pairwise promoter dataset. Returns matrix of dummy features.
    
    Parameters
    -----------
    dfDataset : DataFrame
        Dataframe containing promoter data from pairwise dataset
        columnNames = (sequence_1, 35boxstart_1, 10boxstart_1, sequence_2, 35boxstart_2, 10boxstart_2)
        
    seqRegions : tuple,list
        List containing two positional ranges from which features are derived, respectively starting from first
        nucleotide of 35-box and 10-box
        Example: [[0,6],[0,6]] returns positional features of the range of the -35box and -10box respectively
        
    shuffle : Boolean
        Shuffles input dataset
        
    merge : Boolean
        subtract dummy features of pairwise sequences. Reduces returned features by half
        
    Returns
    --------
    dfPositionBox : DataFrame 
        Dataframe containing dummy arguments
    dfDataset : DataFrame
        Shuffled dataset
    
    """    
    
    dfDataset['sequence_1'] = dfDataset['sequence_1'].str.upper()
    dfDataset['sequence_2'] = dfDataset['sequence_2'].str.upper()
    
    if shuffle is True:
        dfDataset = dfDataset.reindex(np.random.permutation(dfDataset.index))
    
    dfDataset1 = pd.DataFrame(dfDataset[['ID_1','sequence_1','mean_score_1','35boxstart_1','10boxstart_1']].values,columns=['ID','sequence','mean_score','35boxstart','10boxstart'])
    dfDataset2 = pd.DataFrame(dfDataset[['ID_2','sequence_2','mean_score_2','35boxstart_2','10boxstart_2']].values,columns=['ID','sequence','mean_score','35boxstart','10boxstart'])

    
    dfPositionBoxSeq1 = PositionalFeatures(dfDataset1, seqRegions)
    dfPositionBoxSeq2 = PositionalFeatures(dfDataset2, seqRegions)
  
    if subtract is True: 
        dfPositionBox = pd.DataFrame(np.subtract(dfPositionBoxSeq1.values,dfPositionBoxSeq2.values),columns=dfPositionBoxSeq1.columns)
    else:
        dfPositionBox = pd.concat([dfPositionBoxSeq1, dfPositionBoxSeq2], axis=1)
   
        

    if shuffle is False:
        return dfPositionBox
    else:
        return dfDataset, dfPositionBox

    

def SelectRegions(sequences, ntRange, referencePoint=None):
    
    """Selects substring or sequence nucleotides 
    
    Parameters
    -----------
    sequences : 1-dimensional numpy array
        numpy array containing an array of sequences (str)    
        
    ntRange : tuple
        range of nucleotides with respect to a reference point
        
   referencePoint: 1-dimensional numpy array
       numpy array containing the positional reference point for each sequence given in 'sequences'
    
    Returns
    --------
    selectedNucleotides : 1-dimensional numpy array 
        numpy array containing the nucleotide fragments from each region
    """
    if referencePoint.all == None:
        selectedNucleotides = [sequences[u][ntRange[0]:ntRange[1]] for u in range(0,len(sequences))]
    else:
        selectedNucleotides = []
        for u in range(0,len(sequences)):
            pre = ntRange[0]+referencePoint[u]
            diff = ntRange[1]+referencePoint[u] -len(sequences[u])
            if pre<0 and diff>0:
                nucleotide = str(abs(pre)*'-')+sequences[u][ntRange[0]+referencePoint[u]-pre:
                                                            ntRange[1]+referencePoint[u]-diff] +diff*'-'
            elif pre<0 and diff<=0:
                nucleotide = str(abs(pre)*'-')+sequences[u][ntRange[0]+referencePoint[u]-pre:
                                                            ntRange[1]+referencePoint[u]]
            elif pre>=0 and diff>0:
                nucleotide = sequences[u][ntRange[0]+referencePoint[u]:ntRange[1]+referencePoint[u]-diff] +diff*'-'
            elif pre>=0 and diff<=0:
                nucleotide = sequences[u][ntRange[0]+referencePoint[u]:ntRange[1]+referencePoint[u]]
                
            selectedNucleotides.append(nucleotide)                             
    return selectedNucleotides