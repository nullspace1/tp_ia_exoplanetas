
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
        df["duration_error"] = (df["koi_duration_err1"].abs() + df["koi_duration_err2"].abs()) / 2
        df["impact_error"] = (df["koi_impact_err1"].abs() + df["koi_impact_err2"].abs()) / 2
        
        df = df[["kepid", "koi_period","koi_duration","period_error", "duration_error"]]    
        df.rename(columns={
            "kepid": "id",
            "koi_period": "period",
            "koi_duration": "duration",
            "koi_disposition": "has_exoplanet"
        }, inplace=True)

        
        return df[df["period"] < self.period_filter]
    
    def get_distribution(self,kepler_id : str ,variable : dict , variable_name : str):
        planets = self.df[self.df["id"] == kepler_id]
        
        weights = np.zeros(variable["bins"])
        values = np.linspace(variable["min"], variable["max"], variable["bins"]) 
        for _, row in planets.iterrows():
            variable_value = row[variable_name]
            variable_error = row[f"{variable_name}_error"]
            
            weights_to_add = np.zeros(variable["bins"])
            
            min_value = np.min((values-variable_value)**2/(variable_error)**2)
            if ((min_value > 100) == np.True_):
                close_value = np.min(np.abs(values - variable_value))
                weights_to_add= np.exp(-(values-variable_value)**2/(close_value)**2)
            else:
                weights_to_add = np.exp(-(values-variable_value)**2/(variable_error)**2)

            weights_to_add = weights_to_add/np.sum(weights_to_add)
            weights = np.minimum(1, weights + weights_to_add)
        
        return weights
    
    def get_period_distribution(self, kepler_id: str) -> ndarray:
        return self.get_distribution(kepler_id, self.period, "period")
    
    def get_duration_distribution(self, kepler_id: str) -> ndarray:
        return self.get_distribution(kepler_id, self.duration, "duration")
        