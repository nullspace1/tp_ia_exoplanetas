
import os
import zipfile
import kagglehub
from numpy import ndarray
import pandas as pd
import numpy as np

from StarDataset import StarDataset

class Positives(StarDataset):
 
    def __init__(self,config):
        self.period_filter = config["max_period_length"]
        StarDataset.__init__(self,config,"positive")
        self.sample_count = config["positive_sample_count"]
          
    def get_dataset(self, config) -> pd.DataFrame:
        os.environ["KAGGLE_USERNAME"] = config["username"]
        os.environ["KAGGLE_KEY"] = config["key"]
        
        zip_path = kagglehub.dataset_download(
                handle=config["dataset_name"], 
                path="cumulative.csv"
            )
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open("cumulative.csv") as f:
                df = pd.read_csv(f)

        df["period_error"] = (df["koi_period_err1"].abs() + df["koi_period_err2"].abs()) / 2
        
        df = df[["kepid", "koi_period","period_error"]]    
        df.rename(columns={
            "kepid": "id",
            "koi_period": "period",
            "koi_disposition": "has_exoplanet"
        }, inplace=True)

        
        return df[df["period"] < self.period_filter]