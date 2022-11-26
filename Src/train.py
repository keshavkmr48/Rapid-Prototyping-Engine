from sklearn import preprocessing
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes

import category_encoders as ce
import pandas as pd
import argparse
import os

random_state=42

parser = argparse.ArgumentParser(description='trains specified model on cross-validation dataset')

parser.add_argument('-id_cols', nargs='+', action="append",
                    help='speicfy id and id like column in the dataset such that those can be dropped while model training.')
parser.add_argument('-target_col', metavar='TARGET', type=str, nargs='?',default='target',
                    help='specify target column of the datset')
parser.add_argument('-label_encoding', metavar='LE', type=bool, nargs='?',default=True,
                    help='specify whether you want to perform label encoding for categorical columns with atomicity less than 6 or not.')                


parser.add_argument('-labelencoding_cols', nargs='*', action="append",
                    help='speicfy columns on which label encoding should be applied. \
    Please ensure you are understand the difference between nominal and ordinal variables. \
    Also label encoder works best with atomicity less than 10.\
    Avoid label encoder with nominal variables with parametric models or distance based cost functions because label encoding assumes inherent order in variable values.')

parser.add_argument('-target_encoding', metavar='TE', type=bool, nargs='?',default=True,
                    help='specify whether you want to perform target encoding for categorical columns with atomicity more than 10 or not.')                

parser.add_argument('-ohe', metavar='OHE', type=bool, nargs='?',default=True,
                    help='specify whether you want to perform one-hot encoding for categorical columns.')                


parser.add_argument('-standardization', metavar='STNDZN', type=bool, nargs='?',default=True,
                    help='specify whether you want to perform standardization for continuous values.Not required for tree based models.')                


args = parser.parse_args()
id_col= 'id' if args.id_cols is None else args.id_cols[0]
target_col= args.target_col
label_encoding=args.label_encoding
target_encoding=args.target_encoding
ohe=args.ohe
standardization=args.standardization



labelencoding_cols=args.labelencoding_cols[0]


if labelencoding_cols==None:
    label_enoding=False


if type(id_col)=='list':
    drop_col=id_col+list(target_col)
else:
    drop_col=[id_col,target_col]
drop_col.extend("kfold")




TRAINING_DATA=os.environ.get("TRAINING_DATA")
FOLD=os.environ.get("FOLD")

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

def func_target_encoding(te_col,features,target):
    data=pd.concat([features,target],axis=1)
    

    return te_encoder_class, ohe_enc,
    



if __name__=="__main__":
    df=pd.read_csv(TRAINING_DATA)
    train_df=df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df=df[df.kfold==FOLD]

    y_train=train_df[target_col]
    y_val=valid_df[target_col]

    train_df=train_df.drop(drop_col,axis=1)
    valid_df=valid_df.drop(drop_col,axis=1)

    valid_df=valid_df[train_df.columns]


    #categorical columns
    cat_col=train_df.select_dtypes(include=['object']).columns
    cont_cal=train_df.select_dtypes(include=['int','float']).columns


    # select columns for label encoding
    if label_encoding==True:
        
        le_col=set([col for col in cat_col if train_df[col].nunique()<=10]).intersection(set(labelencoding_cols))
        label_encoders=[]
        for col in le_col:
            lbl=preprocessing.LabelEncoder()
            lbl.fit(train_df[col].values.tolist()+valid_df[col].values.tolist())
            train_df.loc[:,col]=lbl.transform(train_df[col].values.tolist())
            valid_df.loc[:,col]=lbl.transform(valid_df[col].values.tolist())
            label_encoders.append((col,lbl))
    
    # select column for one hot encoding
    if ohe==True:
        ohe_col=set([col for col in cat_col if train_df[col].nunique()<=10]).difference(set(labelencoding_cols))
        ohe_encoders=[]
        for col in ohe_col:
            ohe=preprocessing.OneHotEncoder(drop='first',handle_unknown='infrequent_if_exist',min_frequency=0.1)
            ohe.fit(train_df[col].values.tolist()+valid_df[col].values.tolist())
            train_df.loc[:,ohe.get_feature_names(col)]=ohe.transform(train_df[col].values.tolist())
            valid_df.loc[:,ohe.get_feature_names(col)]=ohe.transform(valid_df[col].values.tolist())
            ohe_encoders.append((col,ohe))


    # select column for target encoding
    if target_encoding==True:
        # if one of the two above is true
        if (ohe or label_encoding):
            te_col=[col for col in cat_col if train_df[col].nunique()>10]
        # if both ohe or label encoding is false
        else:
            te_col=cat_col

        ohe_enc=ce.OneHotEncoder()
        ohe_enc.fit(pd.concat([y_train.astype(str),y_val.astype(str)]))
        y_onehot_train=ohe_enc.transform(y_train.astype(str))
        y_onehot_val=ohe_enc.transform(y_val.astype(str))

        class_names=y_onehot_train.columns

        X_obj=pd.concat([train_df,valid_df])[te_col]
        y_onehot=pd.concat([y_onehot_train,y_onehot_val])

        train_df_original=train_df.drop(te_col,axis=1)
        valid_df_original=valid_df.drop(te_col,axis=1)

        te_encoder_class=[]
        for class_ in class_names:
            te_enc=ce.TargetEncoder(smoothing=0)
            te_enc.fit(X_obj,y_onehot[class_])
            temp_train = te_enc.transform(train_df[te_col])
            temp_val= te_enc.transform(valid_df[te_col])
            temp_train.columns=[str(x)+'_'+str(class_) for x in temp_train.columns]
            temp_val.columns=[str(x)+'_'+str(class_) for x in temp_val.columns]
            train_df=pd.concat([train_df_original,temp_train],axis=1)
            valid_df=pd.concat([valid_df_original,temp_val],axis=1)
            
            te_encoder_class.append((class_,te_enc))

        target_encoding_dct = {'ohe':ohe_enc,'class_names':class_names,'target_encoding_list_class':te_encoder_class}



    # select column for standardization
    if standardization==True:
        standrd_encoders=[]
        for col in cont_cal:
            stndrd = preprocessing.StandardScaler()
            stndrd.fit(train_df[col].values.tolist()+valid_df[col].values.tolist())
            train_df.loc[:,col]=stndrd.transform(train_df[col].values.tolist())
            valid_df.loc[:,col]=stndrd.transform(valid_df[col].values.tolist())
            standrd_encoders.append((col,stndrd))

    



    



