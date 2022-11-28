from sklearn import ensemble,neighbors,tree,naive_bayes,metrics
from . import dispatcher
import os
import joblib

kfold = os.environ.get(KFOLD_SPLIT)
target = os.environ.get(TARGET_COLS)

if "__name__"=="__main__":

    for FOLD in range(kfold):
        PROCESSED_TRAINING_DATA=f'Processed/train/{FOLD}/train_processed.csv'
        train_df=pd.read_csv(PROCESSED_TRAINING_DATA)
        X_train=train_df.drop(target,axis=1)
        y_train= train_df[target]

        PROCESSED_VALID_DATA=f'Processed/valid/{FOLD}/valid_processed.csv'
        valid_df = pd.read_csv(PROCESSED_VALID_DATA)
        X_valid=valid_df.drop(target,axis=1)
        y_val=valid_df[target]

        X_valid=X_valid[X_train.columns]

        
        trained_classifiers={}

        for clf in os.environ.get(MODELS):
            clf_model=dispatcher.CLASSIFIERS[clf]
            clf_model.fit(X_train.values,y_train.values)

            training_prediction=clf_model.predict_proba(X_train.values)
            train_roc_auc_score=metrics.roc_auc_score(y_true=y_train,y_score=training_prediction)
            train_log_loss = metrics.log_loss(y_true=y_train,y_pred=training_prediction)
            
            pred=clf_model.predict_proba(X_valid.values)
            val_roc_auc_score=metrics.roc_auc_score(y_true=y_val,y_score=pred)
            val_log_loss = metrics.log_loss(y_true=y_val,y_pred=pred)
            
            dct={}
            dct['model']=clf_model
            dct['training_prediction']=training_prediction
            dct['train_roc_auc_score']=train_roc_auc_score
            dct['train_log_loss']=train_log_loss
            dct['validation_prediction']=pred
            dct['val_roc_auc_score']=val_roc_auc_score
            dct['val_log_loss']=val_log_loss

            trained_classifiers[clf]=dct
        joblib.dump(trained_classifiers,f'Models/{FOLD}/all_models_classifier_metrics.pkl')



        


