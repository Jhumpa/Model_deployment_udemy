import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pipeline import titanic_pipe
import config



def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([config.TARGET], axis = 1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    
#     print(X_train.shape, X_test.shape)


    # divide train and test

    # fit pipeline
    titanic_pipe.fit(X_train, y_train)
#     class_ = titanic_pipe.predict(X_train)
#     pred = titanic_pipe.predict_proba(X_train)[:,1]
#     print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
#     print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
#     print(X_train.shape, X_test.shape)
#     print(X_train.columns)
    

#     # save pipeline
    joblib.dump(titanic_pipe, config.PIPELINE_NAME)
    

if __name__ == '__main__':
    run_training()
