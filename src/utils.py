import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import Custom_Exception
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
        
        logging.info("the object has been saved")
    
    except Exception as e:
        raise Custom_Exception(e,sys)