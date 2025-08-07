import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import Custom_Exception
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
        
        logging.info("the object has been saved")
    
    except Exception as e:
        raise Custom_Exception(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            
            #model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        
        return report
            
            
            
    except Exception as e:
        raise Custom_Exception(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        logging.info("Object has been loaded")
        
    except Exception as e:
        raise Custom_Exception(e, sys)
