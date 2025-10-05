import os
import random
from flask import request
import numpy as np
import pandas as pd
import lightkurve as lk
import requests
from tqdm import tqdm
import shutil
import json
from astroquery.mast import Observations
from astropy.table import Table
from astropy.io import fits

class StarDataset:
    
    df : pd.DataFrame
    sample_count: int
    lightcurve_length: int
    min_lightcurve_length: int
    save_path: str
    period: dict
    segment_map: dict[str, list[int]]
    
    def __init__(self, config, path_key):
        self.df = self.get_dataset(config)
        self.lightcurve_length = config["lightcurve_length"]
        self.min_lightcurve_length = config["min_lightcurve_length"]
        self.save_path = config[path_key]
        self.cache_path = config["cache_path"]
        self.download_path = "data/curves/"
        self.period = {
            "min": config["distribution_params"]["period"]["min"],
            "max": config["distribution_params"]["period"]["max"],
            "bins": config["distribution_params"]["period"]["bins"]
        }
        self.segment_map_path = self.save_path + "/segment_map.json" 
        if not os.path.exists(self.segment_map_path):
            self.segment_map = {}
        else:
            self.segment_map = json.load(open(self.segment_map_path))
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def get_dataset(self, config) -> pd.DataFrame:
        pass
    
    def get_period_distribution(self, kepler_id : str) -> np.ndarray:
        planets = self.df[self.df["id"] == kepler_id]
        
        weights = np.zeros(self.period["bins"])
        values = np.linspace(self.period["min"], self.period["max"], self.period["bins"])
        
        for _, row in planets.iterrows():
            period_value = row["period"]
            period_error = row["period_error"]
            
            weights_to_add = np.zeros(self.period["bins"])
            
            min_value = np.min((values-period_value)**2/(period_error)**2)
            if ((not np.isfinite(min_value)) or ((min_value > 100) == np.True_) or (period_error == 0)):
                close_value = np.min(np.abs(values - period_value))
                weights_to_add = np.exp(-(values-period_value)**2/(close_value)**2)
            else:
                weights_to_add = np.exp(-(values-period_value)**2/(period_error)**2)

            weights_to_add = weights_to_add/np.sum(weights_to_add)
            weights = np.minimum(1, weights + weights_to_add)
        
        return weights

    def get_segment(self, files, kepler_id):
        if len(files)  == 1:
            return 1
        else:
            if (kepler_id not in self.segment_map):
                self.segment_map[kepler_id] = []
            available_segments = [i for i in range(len(files)) if i not in self.segment_map[kepler_id]]
            if len(available_segments) == 0:
                return None
            segment = random.choice(available_segments) 
            self.segment_map[kepler_id].append(segment)
            return segment  
    
    def get_raw_lightcurve(self, kepler_id : str) -> lk.lightcurve.LightCurve | None:
        id = f"kplr0{kepler_id}"
        files = [f for f in os.listdir(self.download_path) if f.startswith(id)]
        segment = self.get_segment(files, kepler_id)
        if segment is None:
            return None
        file = files[segment]
        table  = fits.open(f"{self.download_path}/{file}")[1].data
        time = table["TIME"][~np.isnan(table["TIME"])]
        flux = table["PDCSAP_FLUX"][~np.isnan(table["TIME"])]   
        lightcurve = lk.LightCurve(time=time, flux=flux)
        return lightcurve.remove_nans().remove_outliers().normalize()
    
    
    def lightcurve_to_numpy(self, lightcurve : lk.lightcurve.LightCurve) -> np.ndarray:
        time = lightcurve.time.value
        flux_in = lightcurve.flux.value
        
        flux = np.full(self.lightcurve_length, np.nan)
        median_time_diff = np.median(np.diff(time))

        t0 = time.min()
        available_indices = np.floor((time - t0) / median_time_diff).astype(np.int64)
        np.clip(available_indices, 0, self.lightcurve_length - 1, out=available_indices)

        flux[available_indices] = flux_in

        avg = np.nanmean(flux)
        flux = np.nan_to_num(flux, nan=avg)
        return flux
    
    def save_array(self,kepler_id : str, data : tuple):
        os.makedirs(self.save_path, exist_ok=True)
        base_path = f"{self.save_path}/{kepler_id}_{self.segment_map[kepler_id][-1]}.npz"
        np.savez(base_path, *data)
        
    def clean_cache(self) -> None:
        try:
            cache_path = self.cache_path
            
            if not os.path.exists(cache_path):
                return
                
            for item in os.listdir(cache_path):
                item_path = os.path.join(cache_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                        
        except Exception as e:
            print(f"[WARN] Cache cleanup failed: {e}")
            pass
            
    def download_all_curves(self, kepler_ids : list[str]) -> None:
        
        obs : list[requests.Response] = Observations.query_criteria(
            obs_collection="Kepler",
            target_name=[f"kplr0{kepler_id}" for kepler_id in kepler_ids],
            dataproduct_type="timeseries"
        )

        products = Observations.get_unique_product_list(obs)
        
        filtered = Observations.filter_products(
            products,
            extension=".fits",
            productSubGroupDescription="LLC",
            mrp_only=False
        )
        
        Observations.download_products(
            filtered,
            download_dir=self.download_path,
            mrp_only=False,
            flat=True
        )
    
    def download_data(self) -> None:
        
        kepler_ids = self.df["id"].sample(n=self.sample_count, replace=True, random_state=42).tolist()
        
        kepler_ids_to_process = []
        
        for kepler_id in kepler_ids:
            file_path = f"{self.save_path}/{kepler_id}.npz"
            if not os.path.exists(file_path):
                kepler_ids_to_process.append(kepler_id)
        
        print(f"Downloading {len(kepler_ids_to_process)} lightcurves...")
        
        self.download_all_curves(kepler_ids_to_process)
        
        for kepler_id in tqdm(kepler_ids_to_process, desc="Processing Lightcurves", unit="file"):
            lightcurve = self.get_raw_lightcurve(kepler_id)
            if lightcurve is None:
                continue
            lightcurve_array = self.lightcurve_to_numpy(lightcurve)
            period_distribution = self.get_period_distribution(kepler_id)
            self.save_array(kepler_id, (lightcurve_array, period_distribution))
        
        
            