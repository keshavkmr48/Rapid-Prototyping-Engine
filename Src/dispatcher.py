from sklearn import ensemble,neighbors,tree,naive_bayes,metrics
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import argparse, os


CLASSIFIERS={
        'randomforest': ensemble.RandomForestClassifier(),
        'extratrees': ensemble.ExtraTreesClassifier(),
        'xgboost': XGBClassifier(),
        'lighgbm': LGBMClassifier(),
        'decisiontree': tree.DecisionTreeClassifier(),
        'knn': neighbors.KNeighborsClassifier(),
        'naivebayes': naive_bayes.MultinomialNB()
    }


