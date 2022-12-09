import fnmatch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import heapq
import sklearn
import metrics
import prep
import scipy
import tensorflow as tf

#Loads the results matching fileBase in dirName
def loadResults(dirName, fileBase, model=False):
    files = fnmatch.filter(os.listdir(dirName), fileBase)
    files.sort()
    
    results = []
    for f in files:
        print(f)
        if not model:
            results.append(pickle.load(open("%s/%s"%(dirName, f), "rb")))
        else:
            results.append(tf.keras.models.load_model("%s/%s"%(dirName, f)))

    return results

#Displays graphs of the valiation and training metric over time
#Also returns the average value of metric in each        
def visualizeExperiment(dirName, fileBase, metric='categorical_accuracy'):
    results = loadResults(dirName, fileBase)

    for i, r in enumerate(results):
        if hasattr(r, 'args'):
            print(r['args'])
        plt.plot(r['history'][metric], label='Model {:d}'.format(i+1))
    plt.title('Training')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(loc='lower right', prop={'size': 10})
    #Need to save figure because the current backend is 'agg', plt.show() does not work
    plt.savefig('TrainingExperiment.png')
    plt.show()

    for i, r in enumerate(results):
        #if np.average(heapq.nlargest(10, r['history']['val_' + metric])) > .6: #Leaves only the really good models
        plt.plot(r['history']['val_' + metric], label='Model {:d}'.format(i+1))
    plt.title('Validation')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(loc='lower right', prop={'size': 10})
    plt.savefig('ValidationExperiment.png')
    plt.show()

    accuracy = 0
    for r in results:
        #uses the average of the top 10 accuracies in a result as its accuracy
        accuracy += np.average(heapq.nlargest(10, r['history']['val_' + metric]))
    print('Average Val Accuracy: ', (accuracy/len(results)))

#Displays the confusion matrix
def visualizeConfusion(dirName, fileBase, types = ['validation'], plot=True):
    results = loadResults(dirName, fileBase)
    accuracy = [0] * len(types)
    for r in results:
        
        if hasattr(r['args'], 'rot'): #Did this for backward compatibility
            print('Rotation: ', r['args'].rot)
            
        i = 0
        for t in types:
            key_predict = 'predict_' + t
            key_true = 'true_' + t

            try:
                print(t.capitalize(), ' Accuracy: ', round(r[key_predict+'_eval'][1],3))
                accuracy[i] += r[key_predict+'_eval'][1]
                preds = r[key_predict]
                trues = r[key_true]
                metrics.generate_confusion_matrix(trues, preds, ['28', '30', '31', '32', '33'], plot)
            except KeyError as e:
                print('Error, cannot find key ', t)
            i+=1
    for t in range(0, len(types)):
        print(f'Average {types[t]} accuracy: {accuracy[t]/len(results)}')

def _getScores(model, args, minSubjNum = 0, maxSubjNum = 10000, valid=False):
    if valid:
        rot = args.rot-1
        if args.rot == 0:
            rot=5
    else: #assumes test
        rot = args.rot
    fileName = 'fold%d.csv'%rot

    #Gets the ins and outs without breaking up trials
    ins, outs = prep.parsePropulsionCSV(args.foldsPath, fileName, pklDirectory = '/ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics', breakUpValues = False, printStats = False, minSubjNum=minSubjNum,maxSubjNum=maxSubjNum)

    #Converts the out category to the propulsion score
    for o in range(0,len(outs)):
        for t in range(0, len(outs[o])):
            if(outs[o][t] == -1):
                outs[o][t] = 0
            elif(outs[o][t] == 28):
                outs[o][t] = -2
            else:
                outs[o][t] -= 29 #Setting 29 to 0
    
    #Gets the predictions and converts them to the proper propulsion score
    #Assumes the model does not try to predict 29
    preds = []
    for i in range(0,len(ins)):
        trial = np.split(ins[i],10)
        trial = np.array(trial)
        p = model.predict(trial).argmax(axis=-1)
        for j in range(0, len(p)):
            if outs[i][j] == 0:
                p[j] = 0
            elif p[j] == 0:
                p[j] = -2
        preds.append(p)
    
    #Gets the scaled 5 minute mastery of propulsion score
    trueScores = []
    predScores = []
    for i in range(0, len(preds)):
        coef = 0
        if outs[i].count(0) < 10:
            coef = 10 / (10 - outs[i].count(0))
        trueScores.append(sum(outs[i]) * coef)
        predScores.append(sum(preds[i]) * coef)
        if(trueScores[len(trueScores)-1] < -20):
            print(trueScores[len(trueScores)-1])
            print(outs[i])
    
    return trueScores, predScores

#Loads in models and the args used to create those models
#Assumes the models were created with the format I setup for the tuner
#Split should be either 'rot' or 'cp'
def analyzeFullPredictions(dirName, modelBase, argBase, split = 'rot', valid=False):
    print('Loaded in models:')
    models = loadResults(dirName, modelBase, model=True)
    args = loadResults(dirName, argBase)
    print()

    #Gets the 5 minute mastery of propulsion scores and the predicted scores for each trial
    if split == 'rot':
        trueScores = []
        predScores = []   
    elif split == 'cp':
        trueScores = [[],[]]
        predScores = [[],[]]
    else: #After this, the code knows that split is either cp or rot
        print('Error, split must be "rot" or "cp"')
        return -1

    for i in range(0, len(models)):
        if split=='rot':
            t, p = _getScores(models[i], args[i]['args'], valid=valid)
            trueScores.append(t)
            predScores.append(p)
        elif split == 'cp':
            #Babies without cp
            t, p = _getScores(models[i], args[i]['args'], maxSubjNum = 99, valid=valid)
            for j in range(0,len(t)):
                trueScores[0].append(t[j])
                predScores[0].append(p[j])
            
            #Babies with cp
            t, p = _getScores(models[i], args[i]['args'], minSubjNum = 100, valid=valid)
            for j in range(0,len(t)):
                trueScores[1].append(t[j])
                predScores[1].append(p[j])

    #Flattens them out for the statistics that look accross all 6 rotations
    flatTrues = [t for trues in trueScores for t in trues]
    flatPreds = [p for preds in predScores for p in preds]

    #Calculates the total difference between the true and predicted scores, the absolute difference and the rss
    diffs = []
    absDiffs = []
    rss = 0
    for i in range(0, len(flatTrues)):        
        diffs.append(flatTrues[i] - flatPreds[i])
        absDiffs.append(abs(flatTrues[i] - flatPreds[i]))
        rss += (flatTrues[i] - flatPreds[i]) ** 2
        
    mean = sum(diffs) / len(diffs) 
    print("Mean Difference: ", mean)

    meanerr = sum(absDiffs) / len(absDiffs)
    print("Mean Error: ", meanerr)

    #Calculates the FVAF as 1-mse/var where unbiased variance is used
    mse = rss / len(flatTrues)
    tss = 0
    trueMean = sum(flatTrues) / len(flatTrues)
    for i in flatTrues:
        tss += (i - trueMean) ** 2
    var = tss / (len(flatTrues) - 1)
    print("FVAF: ", 1 - (mse / var)) 
    print()

    cor, pval = scipy.stats.pearsonr(flatTrues, flatPreds)
    print("Complete p value: ", pval)
    print("Complete R^2: ", cor**2)
    #Gets the pearson p value and the correlation and the scatter plot for each model
    for i in range(0, len(trueScores)):
        if split == 'rot':
            label = "Model %d"%(i+1)
        else:
            if i == 0:
                label = "Typical"
            else:
                label = "At Risk"
        plt.scatter(trueScores[i],predScores[i], label = label, s=16)
        
        cor, pval = scipy.stats.pearsonr(trueScores[i], predScores[i])
        print(label)
        print("p value: ", pval)
        print("R^2: ", cor**2)
        print()

    #Creates the rest of the plot
    plt.plot(flatTrues,flatTrues, label = "ideal", color='black')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.axis('equal')
    plt.legend()
    plt.savefig('FullAnalysisTest.png')


        
if __name__=='__main__':
    #visualizeExperiment('results', '*.pkl')
    visualizeConfusion('results', '.pkl', types = ['testing'])
    #analyzeFullPredictions('results', '*model', '*.pkl')
