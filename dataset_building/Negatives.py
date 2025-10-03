import numpy as np
from StarDataset import StarDataset
import pandas as pd

class Negatives(StarDataset):
    def __init__(self,config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["negative_sample_count"]
        StarDataset.__init__(self,config,"negative") 
        
    def get_dataset(self, config) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, sep="|", header=0)
        df = df[df["nconfp"] == 0]
        df = df[["kepid"]]
        df.rename(columns={
        "kepid": "id",
        }, inplace=True)
        return df
        
    def get_period_distribution(self, kepler_id : str):
        return np.zeros(self.period["bins"])
        