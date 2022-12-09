# MOCS-Prediction-Research
The purpose of this research is to improve upon the previous model of predicting the Mastery of Propulsion MOCS scores by Reza Torbati by reducing the number of hyperparameters used.

## Files
The main files that are used during the running of the experiments are `prep.py`, `tunerExperiment.py`, and `tunerModelIdea.py`. <br>
The files `metrics.py` and `post.py` are used for generation of plots and graphs of the results. <br>
The files `tuner.sbatch` and `analyze.sbatch` are used for running the experiments and performing the full analysis of the results, respectively.

## Training
Note: These files are meant to be run on the OU supercomputer. The kinematics data that is used as the input for the models cannot be accessed otherwise.

### Starting the Program
To run the program locally, simply run `tuner.sbatch` as an executable file. This is not recommended due to the time required to tune and run the experiments. <br>
To run the program in the supercomputer, use the command `sbatch tuner.sbatch` to submit it to Slurm to be scheduled to run.
Be sure to make a `results` directory in the main directory before running the file, to avoid errors.

### Arguments
The batch file has several arguments that can be set to modify how the tuner and program will run. More information about these arguments can be found in `tunerExperiment.py`.

#### pklDir
The path to the directory with the kinematic pkl files.

#### exp
This is used by the models as part of the names of the directories and files that are generated by it.

#### foldsPath
The path to the directory with the folds to be used.

#### rot
Determines which rotation of the data is used.

#### epochs
Used to determine the number of epochs to train the final model. Not used for the tuning process.

#### patience
The patience the final model will use. Not used for the tuning process.

#### trials
The number of trials that the tuner will run for.

#### tune
Set to 0 or 1, like a boolean value. Determines whether the tuner should run.

## Analysis
To generate the plots training plots and confusion matrices, simply run `python post.py` while the line to run the full analysis is commented out. Change the arguments used the functions directly in the `post.py` file if desired. <br>
To generate the full analysis and scatter plot, comment out the lines to run the training plots and confusion matrices, then uncomment the line to run the full analysis. Run this file via the `analyze.sbatch` file in Slurm, not locally.
