import numpy as np
import pandas as pd
import os
import sys
import pickle
import math
import skimage.measure

#Returns a list of all of the trials specified in df
def loadTrials(directory, df):
    trials = [] #List of trials to be returned
    df = df.reset_index()
    for index, row in df.iterrows():
        fname = directory + '/' + row['File']
        fname = os.path.splitext(fname)[0]+'.pkl' 
        trial = row['Trial']-1 #This is the trial number that was analyzed
        pkl = pickle.load(open(fname, 'rb'))
        trials.append(pkl[trial]) #Just takes the trial number of interest and appends it to trials
    return trials

'''
Extracts (in this order):
back.pos_filtered (3 values)
cartesian.rot_vel (1 value)
cartesian.vel (3 values)
ft.tared (6 values)
l_ankle.pos_filtered (3 values)
l_elbow.pos_filtered (3 values)
l_foot.pos_filtered (3 values)
l_knee.pos_filtered (3 values)
l_shoulder.pos_filtered (3 values)
l_wrist.pos_filtered (3 values)
r_ankle.pos_filtered (3 values)
r_elbow.pos_filtered (3 values)
r_foot.pos_filtered (3 values)
r_knee.pos_filtered (3 values)
r_shoulder.pos_filtered (3 values)
r_wrist.pos_filtered (3 values)
sippc_action.action as two groups of 1 hot encodings:
    1st is either no movement, powersteering or suit
    2nd is left, right, forward or back
'''
def parseTrials(trialData, reduceSize = 1):
    data = []
    for t in trialData:
        array = t['back']['pos_filtered']
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.array(array)[:15000] #Sometimes there are more than 15000 time steps so just take the first 15000

        array = t['cartesian']['rot_vel']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['cartesian']['vel']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['ft']['tared']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0,0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['l_ankle']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)
        
        array = t['l_elbow']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['l_foot']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['l_knee']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['l_shoulder']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['l_wrist']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['r_ankle']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)
        
        array = t['r_elbow']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['r_foot']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['r_knee']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['r_shoulder']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['r_wrist']['pos_filtered']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0,0,0]], axis = 0)
        exampleData = np.concatenate((exampleData,array),axis = 1)

        array = t['sippc_action']['actions']
        array = np.array(array)[:15000]
        while(len(array) < 15000):
            array = np.append(array, [[0]], axis = 0)
        
        encoding1 = array#This is a one hot encoding with 0: no movement, 1: power steering, 2: suit
        encoding2 = array#This is a one hot encoding with 0: no movement, 1: forward, 2: backward, 3: left, 4: right
        for i in range(0,15000):
            encoding1[i] = math.ceil(encoding1[i] / 4.0)
            encoding2[i] = encoding2[i] % 4
            if(encoding2[i] == 0 and array[i] > 0):
                encoding2[i] = 4
        exampleData = np.concatenate((exampleData,encoding1),axis = 1)
        exampleData = np.concatenate((exampleData,encoding2),axis = 1)

        #Reduces to 7500 time steps or 750 time steps per label
        if reduceSize > 1:
            exampleData = skimage.measure.block_reduce(exampleData, (reduceSize,1), np.average)

        data.append(exampleData)
    
    return data

def translateScore(score):
    if score == 28:
        return -2
    elif score == 29:
        return -1
    elif score == -1:
        return 0
    elif score == 30:
        return 1
    elif score == 31:
        return 2
    elif score == 32:
        return 3
    elif score == 33:
        return 4
    else:
        print("Error, invalid score")

def getFinalScore(row):
    score = 0
    score += translateScore(row['0'])
    score += translateScore(row['30'])
    score += translateScore(row['60'])
    score += translateScore(row['90'])
    score += translateScore(row['120'])
    score += translateScore(row['150'])
    score += translateScore(row['180'])
    score += translateScore(row['210'])
    score += translateScore(row['240'])
    score += translateScore(row['270'])
    return score

def getStats(df):
    freqs = [0] * 6
    times = ['0', '30', '60', '90', '120', '150', '180', '210', '240', '270']
    for index, row in df.iterrows():
        #if getFinalScore(row) != row['FinalScore']:
            #print("WARNING: score mismatch in row ", index+2)
            #print(f"Subject {row['Subject']} on {row['Date']} expected {getFinalScore(row)}, got {row['FinalScore']}")
        for t in times:
            if row[t]-28 >=0:
                freqs[row[t]-28]+=1
    
    for i in range(0,len(freqs)):
        print(f'{i+28}: {freqs[i]}, {(freqs[i] / sum(freqs)) * 100}%')
    print()

#Returns a list of trials as described in pklDictionary
#Inside each trial is an array containing 15000 time steps
#Instead each time step is the list of 51 extracted values show above
def parsePropulsionCSV(directory, fname, pklDirectory = '', reduceSize=1, breakUpValues = True, printStats=True, nogo=False, minSubjNum = 0, maxSubjNum=10000):
    if pklDirectory == '':
        pklDirectory = './KinPklData'

    fname = directory + '/' + fname
    df = pd.read_csv(fname)
    df = df[df['Subject'] >= minSubjNum]
    df = df[df['Subject'] <= maxSubjNum]

    if printStats:
        getStats(df)
    if nogo:
        return -1

    labels = df[['0', '30', '60', '90', '120', '150', '180', '210', '240', '270']].values.tolist()

    trialData = loadTrials(pklDirectory, df)    
    data = parseTrials(trialData, reduceSize)

    if not breakUpValues:
        return data , labels 

    splitData = []
    for i in data:
        splitData += np.split(i, 10)
    
    splitLabels = [l for label in labels for l in label]
    
    '''
    print(len(splitData))
    print(len(splitData[0]))
    print(len(splitData[0][0]))
    print(len(splitLabels))
    '''

    return splitData, splitLabels
        
if __name__=='__main__':
    parsePropulsionCSV('.', sys.argv[1], printStats=True, nogo=True)
    #parsePropulsionCSV('.', 'MasteryOfPropulsionData.csv', printStats=True, nogo=True)
    #print()
    #parsePropulsionCSV('.', 'MasteryOfPropulsionTrain.csv', printStats=True, nogo=True)
    #print()
    #parsePropulsionCSV('.', 'MasteryOfPropulsionValid.csv', printStats=True, nogo=True)
    #parsePropulsionCSV('.', 'TestPropulsion.csv')
