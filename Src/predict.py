from sklearn import ensemble,neighbors,tree,naive_bayes,metrics
from . import dispatcher
import os
import joblib
import pandas as pd

kfold = int(os.environ.get("KFOLD_SPLIT"))
target = os.environ.get("TARGET_COLS")

TEST_DIRECTORY=os.environ.get("TEST_DIRECTORY")
TEST_DATA=os.environ.get("TEST_DATA")
test_data_path = os.path.join(TEST_DIRECTORY,TEST_DATA)


column_dct=joblib.load('Processed/processed_columns_dct.pkl')


for FOLD in range(kfold):
    df = pd.read_csv(test_data_path)

    # X_test=df.values

    #preprocessing pipeline
    for item in [item for item in os.listdir(f'./Models/{FOLD}') if item.endswith('.pkl')]:
        temp=item.split('_')[2]

        if temp=='ohe':
            ohe_col=column_dct['ohe_encoding']
            ohe_encoders = joblib.load('Models/{FOLD}/{item}')

            assert len(ohe_col)==len(ohe_encoders)

            for col,ohe_enc,new_columns in ohe_encoders:
                temp_data=ohe_enc.transform(np.array(df[col].values.tolist()).reshape(-1,1)).toarray()
                df.loc[:,new_columns]=temp_data
                df.drop(col,axis=1,inplace=True)
            

        if temp=='label':
            le_col=column_dct['label_encoding']
            label_encoders = joblib.load('Models/{FOLD}/{item}')

            assert len(le_col)==len(label_encoders)

            for col,le_enc in label_encoders:
                df.loc[:,col]=le_enc.transform(df[col].values.tolist())


        if temp=='target':
            te_col=column_dct['target_encoding']
            te_encoder_class = joblib.load('Models/{FOLD}/{item}')

            df_original=df.drop(te_col,axis=1)

            for class_,te_enc,col_name in te_encoder_class:
                temp = te_enc.transform(df[te_col])
                temp.columns=col_name
                df_original=pd.concat([df_original,temp],axis=1)

            df=df_original

        if temp=='stndrd':
            cont_col=column_dct['stndrd_encoding']
            standrd_encoders= joblib.load('Models/{FOLD}/{item}')

            assert len(cont_col) ==len(standrd_encoders)

            for col,stndrd_enc in standrd_encoders:
                df.loc[:,col]=stndrd_enc.transform(df[col].values.tolist())


    # classifier prediction pipeline

    


#final prediction : average of all folds















