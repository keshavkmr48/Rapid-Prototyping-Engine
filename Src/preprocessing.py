from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import category_encoders as ce
import pandas as pd
import argparse
import os
import numpy as np
import joblib

random_state=42


id_col= os.environ.get("ID_COLS")
target_col= os.environ.get("TARGET_COLS")
# label_encoding=os.environ.get("LABEL_ENCODING_FLAG")
label_encoding=os.getenv("LABEL_ENCODING_FLAG", 'False').lower() in ('true', '1', 't')
# target_encoding=os.environ.get("TARGET_ENCODING_FLAG")
target_encoding=os.getenv("TARGET_ENCODING_FLAG", 'False').lower() in ('true', '1', 't')
# ohe=os.environ.get("OHE_FLAG")
ohe=os.getenv("OHE_FLAG", 'False').lower() in ('true', '1', 't')
# standardization=os.environ.get("STANDARDIZATION_FLAG")
standardization=os.getenv("STANDARDIZATION_FLAG", 'False').lower() in ('true', '1', 't')

labelencoding_cols=os.environ.get("LABEL_ENCODING_COLUMNS")

print('label_encoding',label_encoding)
print('ohe_encoding',ohe)
print('target_encoding',target_encoding)
print('standardization',standardization)

if labelencoding_cols==None:
    labelencoding_cols=''


if isinstance(id_col,list):
    id_col.extend([target_col])
    drop_col=id_col
else:
    drop_col=[id_col,target_col]
drop_col.extend(["kfold"])




TRAINING_DATA=os.environ.get("TRAINING_DATA")
kfold = int(os.environ.get("KFOLD_SPLIT"))


FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

TRAINING_DIRECTORY=os.environ.get("TRAINING_DIRECTORY")
FOLDED_DATA=os.environ.get("FOLDED_DATA")
training_folded_data_path = os.path.join(TRAINING_DIRECTORY,FOLDED_DATA)
    



if __name__=="__main__":
    print('preprocessing started...')

    for FOLD in range(kfold):
        df=pd.read_csv(training_folded_data_path)


        print(f'{FOLD} STARTED ...')

        if not os.path.exists(f'Models/{FOLD}'):
            os.makedirs(f'Models/{FOLD}')

        train_df=df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
        valid_df=df[df.kfold==FOLD]

        y_train=train_df[target_col]
        y_val=valid_df[target_col]
        train_df=train_df.drop(drop_col,axis=1)
        valid_df=valid_df.drop(drop_col,axis=1)

        valid_df=valid_df[train_df.columns]
        print(train_df.shape)


        #categorical columns
        cat_col=train_df.select_dtypes(include=['object']).columns
        cont_cal=train_df.select_dtypes(include=['int','float']).columns
        # print('cat_col',cat_col)
        # # print(set(labelencoding_cols))
        # print(labelencoding_cols)

        # select columns for label encoding
        if label_encoding==True:
            if ohe==False:
                le_col=[col for col in cat_col if train_df[col].nunique()<=10]
            else:
                le_col=list(set([col for col in cat_col if train_df[col].nunique()<=10]).intersection(set(labelencoding_cols)))
            if len(le_col)>0:
                # print('label_encoding started...',le_col)
                label_encoders=[]
                for col in le_col:
                    lbl=preprocessing.LabelEncoder()
                    lbl.fit(train_df[col].values.tolist()+valid_df[col].values.tolist())
                    train_df.loc[:,col]=lbl.transform(train_df[col].values.tolist())
                    valid_df.loc[:,col]=lbl.transform(valid_df[col].values.tolist())
                    label_encoders.append((col,lbl))
                joblib.dump(label_encoders,f'Models/{FOLD}/all_col_label_encoders.pkl')
                print('label_encoding_completed', train_df.shape)
                
            
        # select column for one hot encoding
        if ohe==True:
            ohe_col=list(set([col for col in cat_col if train_df[col].nunique()<=10]).difference(set(labelencoding_cols)))
            print(ohe_col)
            ohe_encoders=[]
            for col in ohe_col:
                ohe_enc=preprocessing.OneHotEncoder(drop='first',handle_unknown='infrequent_if_exist',min_frequency=0.1)
                ohe_enc.fit(np.array(train_df[col].values.tolist()+valid_df[col].values.tolist()).reshape(-1,1))
                new_columns=ohe_enc.get_feature_names_out([col])
                temp_data_train=ohe_enc.transform(np.array(train_df[col].values.tolist()).reshape(-1,1)).toarray()
                temp_data_val=ohe_enc.transform(np.array(valid_df[col].values.tolist()).reshape(-1,1)).toarray()
                # print(temp_data_train.shape,new_columns)
                # print(temp_data_val.shape,new_columns)
                # print(ohe.transform(np.array(train_df[col].values.tolist()).reshape(-1,1)).shape)
                train_df.loc[:,new_columns]=temp_data_train
                valid_df.loc[:,new_columns]=temp_data_val
                ohe_encoders.append((col,ohe_enc,new_columns))

            joblib.dump(ohe_encoders,f'Models/{FOLD}/all_col_ohe_encoders.pkl')
            print('ohe_encoding_completed', train_df.shape)

            
            

        # select column for target encoding
        if target_encoding==True:
            # if one of the two above is true
            prcoessed_col = le_col+ohe_col
            if len(prcoessed_col)>1:
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
                col_name=[str(x)+'_'+str(class_) for x in temp_train.columns]
                temp_train.columns=[str(x)+'_'+str(class_) for x in temp_train.columns]
                temp_val.columns=[str(x)+'_'+str(class_) for x in temp_val.columns]
                train_df_original=pd.concat([train_df_original,temp_train],axis=1)
                valid_df_original=pd.concat([valid_df_original,temp_val],axis=1)

                
                te_encoder_class.append((class_,te_enc,col_name))

            # target_encoding_dct = {'ohe':ohe_enc,'class_names':class_names,'target_encoding_list_class':te_encoder_class}

            train_df=train_df_original
            valid_df=valid_df_original
            joblib.dump(te_encoder_class,f'Models/{FOLD}/all_col_target_encoding_dct.pkl')
            print('target_encoding_completed', train_df.shape)



        # select column for standardization
        if standardization:
            standrd_encoders=[]
            for col in cont_cal:
                stndrd = preprocessing.StandardScaler()
                stndrd.fit(np.array(train_df[col].values.tolist()+valid_df[col].values.tolist()).reshape(-1,1))
                train_df.loc[:,col]=stndrd.transform(np.array(train_df[col].values.tolist()).reshape(-1,1))
                valid_df.loc[:,col]=stndrd.transform(np.array(valid_df[col].values.tolist()).reshape(-1,1))
                standrd_encoders.append((col,stndrd))
            joblib.dump(standrd_encoders,f'Models/{FOLD}/all_col_stndrd_encoders.pkl')
            print('standardization_completed', train_df.shape)

        drop_col_post_processing=ohe_col
        # drop_col_post_processing.extend(te_col)

        # print(drop_col_post_processing)
        # print(train_df.columns)
        print('droping columns',drop_col_post_processing)
        train_df.drop(drop_col_post_processing,axis=1,inplace=True)
        valid_df.drop(drop_col_post_processing,axis=1,inplace=True)
        

        if not os.path.exists(f'Processed/train/{FOLD}'):
            os.makedirs(f'Processed/train/{FOLD}')

        if not os.path.exists(f'Processed/valid/{FOLD}'):
             os.makedirs(f'Processed/valid/{FOLD}')

        pd.concat([train_df,y_train],axis=1).to_csv(f'Processed/train/{FOLD}/train_processed.csv',index=False)
        pd.concat([valid_df,y_val],axis=1).to_csv(f'Processed/valid/{FOLD}/valid_processed.csv',index=False)

        column_dct={}
        column_dct['ohe_encoding']=ohe_col
        column_dct['label_encoding']=le_col
        column_dct['target_encoding']=te_col
        column_dct['stndrd_encoding']=cont_cal

        joblib.dump(column_dct,f'Processed/processed_columns_dct.pkl')
    
    print('preprocessing_finished...')




        



    



