import pandas as pd
import numpy as np
import argparse
import pickle
import random
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
import keras_tuner
from keras_tuner.tuners import BayesianOptimization

from prep import parsePropulsionCSV
from tunerModelIdea import build_model

def load_file(path, fname, args):
    ins, outs = parsePropulsionCSV(path, fname, args.pklDir, reduceSize=1) #reducing is now done in the model

    #Removes any bad intervals and subtracts 28 from the value
    i=0
    while i < len(outs):
        if outs[i] == -1 or outs[i] == 29:
            del(outs[i])
            del(ins[i])
            i-=1
        elif outs[i] == 28:
            outs[i] -= 28
        else:
            outs[i] -= 29
        i+=1     
    
    return ins, outs

#Loads in and prepares the data
def load_data(args):
    #I am assuming that there are 6 folds
    #Test fold = rot
    #Validation fold = rot - 1 or 5 if rot==0
    #Train = remaining folds
    
    #Prepares the folds for the training, validation and test data
    train = []
    for i in range(0,6):
        train.append('fold%d.csv'%(i))
    
    test = 'fold%d.csv'%(args.rot)
    del train[args.rot]
    
    if args.rot > 0:
        validation = 'fold%d.csv'%(args.rot-1)
        del train[args.rot-1]
    else:
        validation = 'fold5.csv'
        del train[4] #This is 4 instead of 5 because I've already deleted an element above

    #Loads in the data
    print('Train data:')
    ins, outs = load_file(args.foldsPath, train[0], args)
    for fold in train[1:]:
        tmpIns, tmpOuts = load_file(args.foldsPath, fold, args)
        ins.extend(tmpIns)
        outs.extend(tmpOuts)
    
    print('Validation data:')
    validIns, validOuts = load_file(args.foldsPath, validation, args)
    
    print('Test data:')
    testIns, testOuts = load_file(args.foldsPath, test, args)

    #Converts the data from lists to numpy arrays   
    ins = np.array(ins)
    outs = np.array(outs)
    validIns = np.array(validIns)
    validOuts = np.array(validOuts)
    testIns = np.array(testIns)
    testOuts = np.array(testOuts)

    #One hot encodes the outputs. There are 6 output classes
    outs = np.eye(5)[outs]
    validOuts = np.eye(5)[validOuts]
    testOuts = np.eye(5)[testOuts]
    return ins, outs, validIns, validOuts, testIns, testOuts

#Creates a string with all of the important training metadata to be used for file names
def generate_fname(args, tuner:bool):
    if not tuner:
        return '%s/%s_trials%d_rot%d'%(
            args.resultsPath,
            args.exp,
            args.trials,
            args.rot
        )
    if tuner:
        return '%s/%s_trials%d_rot%d'%(
            args.resultsPath,
            args.exp,
            args.trials,
            args.rot
        )

#Used to generate batches of examples
#inputName needs to match the name of the input layer
#outputName needs to match the name of the output layer
def batch_generator(ins, outs, batchSize, inputName='input', outputName='output'):
    while True:
        #Gets a batchSize sized sample from the inputs
        indicies = random.choices(range(ins.shape[0]), k=batchSize)

        #Returns a list of the selected examples and their corresponding outputs
        yield({inputName: ins[indicies,:,:]}, {outputName: outs[indicies,:]})

def execute_exp(args):
    ins, outs, validIns, validOuts, testIns, testOuts = load_data(args)

    fbase = generate_fname(args, tuner=True) #gets a file name for the tuner
    keras.backend.clear_session()

    #Creates the tuner
    tuner = BayesianOptimization(
        build_model,
        objective = 'val_categorical_accuracy',
        num_initial_points = 60,
        max_trials = args.trials,
        project_name=fbase,
        overwrite=args.overwrite,
        directory=args.logDir
    )
    tuner.search_space_summary()

    if args.tune:
        tunerCallback = keras.callbacks.EarlyStopping(
            patience = 40, 
            restore_best_weights=True, 
            monitor='val_categorical_accuracy',
            mode='max'
        )
        tuner.search(
            ins, outs, 
            epochs = args.epochs, 
            validation_data=(validIns, validOuts),
            callbacks=[tunerCallback, keras.callbacks.TensorBoard(fbase)]
        )

    bestHyper = tuner.get_best_hyperparameters(1)[0]
    print(bestHyper.values)
    model  = tuner.hypermodel.build(bestHyper)

    if args.nogo:
        return

    #Runs the model
    modelCallback = keras.callbacks.EarlyStopping(patience=args.patience,
        restore_best_weights=True,
        monitor='val_categorical_accuracy',
        mode='max',
    )

    history = model.fit(x=ins, y=outs, epochs=args.epochs,
	    verbose=True,
	    validation_data=(validIns, validOuts),
	    callbacks=[modelCallback]
    )
  
    fbase = generate_fname(args, tuner = False) #gets a file name for the model
    # Generate log data
    results = {}
    results['args'] = args
    results['hyperparameters'] = bestHyper
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['true_training'] = outs
    results['predict_validation'] = model.predict(validIns)
    results['predict_validation_eval'] = model.evaluate(validIns, validOuts)
    results['true_validation'] = validOuts
    results['predict_testing'] = model.predict(testIns)
    results['predict_testing_eval'] = model.evaluate(testIns, testOuts)
    results['true_testing'] = testOuts

    results['history'] = history.history

    # Save results
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Model
    model.save("%s_model"%(fbase))
    
    return model

#Create the parser for the command-line arguments
def create_parser():
    parser = argparse.ArgumentParser(description='Mastery of Propulsion Learner', fromfile_prefix_chars='@')
    
    parser.add_argument('-exp', type=str, default='Propulsion', help='Tag to be put in file name')
    parser.add_argument('-rot', type=int, default=1, help='rotation')
    parser.add_argument('-foldsPath', type=str, default='folds', help='Path to the csv files with the folds')
    parser.add_argument('-resultsPath', type=str, default='results', help='Directory to store results in')
    parser.add_argument('-pklDir', type=str, default='', help='Directory to the pkl files')
    parser.add_argument('-logDir', type=str, default='logs', help='Directory of the log files for the tuner')    

    parser.add_argument('-trials', type = int, default = 1, help='Number of trials to run the tuner for')
    parser.add_argument('-overwrite', action = 'count', default = 0, help = 'Overwites the tuner if set')
    parser.add_argument('-tune', type=int, default = 1, help = 'whether or not to actually run the tuner. Used as a bool')

    parser.add_argument('-epochs',type=int, default = 10, help='Number of epochs to run for')
    parser.add_argument('-patience', type=int, default = 100, help='Patience for early termination')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    return parser

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(12)
    parser = create_parser()
    args = parser.parse_args()

    execute_exp(args)
    
