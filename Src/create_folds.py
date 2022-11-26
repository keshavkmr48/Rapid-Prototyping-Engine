
# @author: KeshavKMR48
# Date: 26-11-2022

import pandas as pd
import numpy as np
import argparse 
from sklearn import model_selection

"""
1. need target variable with the name target for supervised learning. Ignore for supervised learning.
2. File name should be train.csv placed at Input/train.csv/ folder.
3. Output file will be available at Input/train.csv/train_folds.csv
"""

parser = argparse.ArgumentParser(description='Creates k-fold data for model training by applying the chosen cross-validation strategy.')

parser.add_argument('n_split', metavar='N', type=int, nargs='?', choices=range(2,11),
                    help='speicfy number of splits for k-fold')
parser.add_argument('cross_validation_strategy', metavar='C', type=str, nargs='?', choices=['StratifiedKFold','KFold'],
                    help='speicfy the cross-validation strategy of choice')

args = parser.parse_args()
n_splits=5 if args.n_split is None else args.n_split
cross_validation_strategy='KFold' if args.cross_validation_strategy is None else args.cross_validation_strategy
random_state=42


if __name__=="__main__":
    df=pd.read_csv("Input/train.csv/train.csv")
    df["kfold"]=-1
    df=df.sample(frac=1).reset_index(drop=True)

    if cross_validation_strategy=='StratifiedKFold':  
        """StratifiedKFold is a variation of k-fold which returns stratified folds: 
        each set contains approximately the same percentage of samples of each target class as the complete set."""

        kf=model_selection.StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)

    elif cross_validation_strategy=='KFold':
        """KFold divides all the samples in k  groups of samples, called folds (if k==n, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). 
        The prediction function is learned using k-1 folds, and the fold left out is used for test."""
        kf=model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=random_state)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
            print(len(train_idx),len(val_idx))
            df.loc[val_idx,"kfold"]=fold
        
    df.to_csv("Input/train.csv/train_folds.csv", index=False)