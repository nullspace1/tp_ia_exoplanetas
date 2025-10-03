from pandas.core.api import DataFrame as DataFrame
from StarDataset import StarDataset
import pandas as pd
import numpy as np
from numpy import nan


class Synthetics(StarDataset):
    def __init__(self, config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["synthetic_sample_count"]
        StarDataset.__init__(self, config, "synthetic")

        
    def get_dataset(self, config) -> DataFrame:
        df = pd.read_csv(self.data_path, sep="|", header=0)
        
        df = df[df["nconfp"] == 0]
        
        df = df[["kepid"]]
        df.rename(columns={
        "kepid": "id",
        }, inplace=True)
        df["has_exoplanet"] = nan
        df["period"] = nan
        df["duration"] = nan
        df["period_error"] = nan
        df["duration_error"] = nan
        
        for _ in range(self.sample_count // 4):
            count = np.random.randint(1, 10)
            new_exoplanets = df.sample(n = count, random_state=42).reset_index(drop=True)
            new_exoplanets["has_exoplanet"] = 1
            new_exoplanets["period"] = np.random.uniform(config["distribution_params"]["period"]["min"], config["distribution_params"]["period"]["max"], count)
            new_exoplanets["duration"] = np.random.uniform(config["distribution_params"]["duration"]["min"], config["distribution_params"]["duration"]["max"], count)
            new_exoplanets["period_error"] = np.random.uniform(0,0.1,count)
            new_exoplanets["duration_error"] = np.random.uniform(0,0.1,count)
            
            df = pd.concat([df, new_exoplanets])

        df = df[df["has_exoplanet"] == 1]
        df = df[["id", "period", "duration", "period_error", "duration_error"]]
        
        
        return df
