## User Input for each experiment
export TRAINING_DATA=train_folds.csv
export MODELS=randomforest,extratrees,xgboost,lighgbm,decisiontree
export TARGET_COLS=target
export KFOLD_SPLIT=5 
export KFOLD_SPLITTING_STRATEGY=StratifiedKFold
export ID_COLS=id
export OHE_FLAG=True
export TARGET_ENCODING_FLAG=True
export LABEL_ENCODING_FLAG=True
export LABEL_ENCODING_COLUMNS='None'
export STANDARDIZATION_FLAG=True
export THRESHOLD=0.5



# Code Constants
export TRAINING_DIRECTORY=Input/train
export FOLDED_DATA=train_folds.csv





# python -m Src.create_folds

# python -m Src.preprocessing

# python -m Src.dispatcher

python -m Src.train






