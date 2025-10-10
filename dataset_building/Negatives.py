from lightkurve.lightcurve import LightCurve
import numpy as np
from StarDataGenerator import StarDataGenerator
import pandas as pd

class Negatives(StarDataGenerator):
    def __init__(self,config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["negative_sample_count"]
        StarDataGenerator.__init__(self,config,"negative") 
        self.download_path = self.download_path + "negative/"
        
    def get_dataset(self, config) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, sep="|", header=0)
        df = df[df["nconfp"] == 0]
        df = df[["kepid"]]
        df.rename(columns={
        "kepid": "id",
        }, inplace=True)
        return df
    
    def get_raw_lightcurve(self, kepler_id: str) -> LightCurve | None:
        return super().get_raw_lightcurve(kepler_id)
        
    def get_period_distribution(self, kepler_id : str):
        return np.zeros(self.period["bins"])
        