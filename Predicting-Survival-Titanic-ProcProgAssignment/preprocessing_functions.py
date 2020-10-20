import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)




def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=0)
    
    return X_train, X_test, y_train, y_test
    



def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0] # captures the first letter
    return df


def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df


    
def impute_na(df, var, replacement="Missing"):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
     return df[var].fillna(replacement)



def remove_rare_labels(df, var, frequent_labels):
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    
    df = df.copy()
    
    # to create the binary variables, we use get_dummies from pandas
    
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
    df.drop(labels=var, axis=1, inplace=True)
    
    return df



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    pass
    

def train_scaler(df, output_path):
    # create scaler
    scaler = StandardScaler()

    #  fit  the scaler to the train set
    scaler.fit(df) 
    joblib.dump(scaler, output_path)


    return scaler
    

def scale_features(df, output_path):
    scaler = joblib.load(output_path) # with joblib probably
    return scaler.transform(df)



def train_model(df, target, output_path):
    model = LogisticRegression(C=0.0005, random_state=0)
    # train the model
    model.fit(df, target)
    # save the model
    joblib.dump(model, output_path)
    return model


def predict(df, model):
    model = joblib.load(model)
    class_ = model.predict(df)
    pred = model.predict_proba(df)[:,1]

    return model.predict(df)

