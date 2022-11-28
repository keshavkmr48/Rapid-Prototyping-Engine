## User Input for each experiment
export TRAINING_DATA='train_folds.csv' #string
export MODELS = 'randomforest','extratrees','xgboost','lighgbm','decisiontree' # list
export TARGET_COLS = 'target' # string
export KFOLD_SPLIT=5 # Range
export KFOLD_SPLITTING_STRATEGY='StratifiedKFold' #option to choose
export ID_COLS = 'id' # list
export OHE_FLAG = true
export TARGET_ENCODING_FLAG = true
export LABEL_ENCODING_FLAG = true
export LABEL_ENCODING_COLUMNS =
export STANDARDIZATION_FLAG= false
export THRESHOLD = 



# Code Constants
export TRAINING_DIRECTORY='Input/train'
export FOLDED_DATA='train_folds.csv'





python Src/create_folds.py

python Src/preprocessing.py 

python Src/dispatcher.py 





