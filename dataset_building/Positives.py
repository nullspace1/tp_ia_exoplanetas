
import os
import zipfile
import kagglehub
from lightkurve.lightcurve import LightCurve
from more_itertools import first
from numpy import ndarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from StarDataGenerator import StarDataGenerator

class Positives(StarDataGenerator):
 
    def __init__(self,config):
        self.period_filter = config["max_period_length"]
        StarDataGenerator.__init__(self,config,"positive")
        self.sample_count = config["positive_sample_count"]
        self.download_path = self.download_path + "positive/"

          
    def get_raw_lightcurve(self, kepler_id: str) -> LightCurve | None:
        lightcurve =  super().get_raw_lightcurve(kepler_id)
        if lightcurve is None:
            return None
        return self.verify_lightcurve(kepler_id, lightcurve)
          
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
        df["first_transit"] = df["koi_time0bk"]
         
        df = df[df["koi_disposition"] == "CONFIRMED"]
        
        df = df[["kepid", "koi_period","period_error", "first_transit"]]    
        df.rename(columns={
            "kepid": "id",
            "koi_period": "period"
        }, inplace=True)

        
        return df