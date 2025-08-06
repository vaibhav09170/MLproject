import os
import sys
import pandas as pd
import numpy as np

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_Feature = ['reading_score', 'writing_score']
            categorical_Feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
                
            )
            logging.info("Numerical column scalling completed")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    
                ]
            )
            
            logging.info("Categorical column enconding completed")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_Feature),
                    ("cat_pipeline", cat_pipeline, categorical_Feature)
                ]
            )
            logging.info("Returning preprocessor object")
            return preprocessor
        
        except Exception as e:
            raise Custom_Exception(e,sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test dataset completed")
            
            logging.info("fetching prepocessing object")
            preprocessor_obj = self.get_data_transformer_object()
            
            
            target_column_name = "math_score"
            numerical_Feature = ['reading_score', 'writing_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessor object on training and testing dataframes")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info("Train fit_transform completed")
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info(f"Applying np.C_")
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object. ")
            
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise Custom_Exception(e,sys)