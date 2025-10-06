import os
import random
import uuid
from flask import request
from matplotlib.style import available
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
    download_count: int
    lightcurve_length: int
    min_lightcurve_length: int
    save_path: str
    period: dict
    sample_count : int
    
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
        self.download_count = config["download_count"]
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


    
    def get_raw_lightcurve(self, kepler_id : str) -> lk.lightcurve.LightCurve | None:
        prefix = f"kplr0{kepler_id}"
        files = [f for f in os.listdir(self.download_path) if f.startswith(prefix)]
        if not files:
            return None

        file = random.choice(files)
        try:
            table = fits.open(f"{self.download_path}/{file}", memmap=False)[1].data
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
            return None
        valid_mask = ~np.isnan(table["TIME"]) & ~np.isnan(table["PDCSAP_FLUX"])

        time = table["TIME"][valid_mask]
        flux = table["PDCSAP_FLUX"][valid_mask]

        if len(time) < self.lightcurve_length:
            return None

        start = np.random.randint(0, len(time) - self.lightcurve_length)
        time = time[start:start + self.lightcurve_length]
        flux = flux[start:start + self.lightcurve_length]

        lc = lk.LightCurve(time=time, flux=flux)
        return lc.remove_nans().remove_outliers().normalize()
    
    
    def lightcurve_to_numpy(self, lightcurve : lk.lightcurve.LightCurve) -> np.ndarray:
        time = lightcurve.time.value
        flux_in = lightcurve.flux.value
        
        flux = np.full(self.lightcurve_length, np.nan)
        median_time_diff = np.median(np.diff(time))

        t0 = time.min()
        available_indices = np.floor((time - t0) / median_time_diff).astype(np.int64)
        
        available_indices = np.clip(available_indices, 0, self.lightcurve_length - 1)

        flux[available_indices] = flux_in

        avg = np.nanmean(flux)
        flux = np.nan_to_num(flux, nan=avg)
        return flux
    
    def save_array(self,kepler_id : str, data : tuple):
        os.makedirs(self.save_path, exist_ok=True)
        base_path = f"{self.save_path}/{kepler_id}_{uuid.uuid4()}.npz"
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
        
                
        print(f"Downloading {len(kepler_ids)} lightcurves...")    
        
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
    
    def get_already_downloaded_ids(self) -> list[str]:
        downloaded_ids = set(int(os.path.splitext(os.path.basename(x))[0].split("-")[0].lstrip("kplr").lstrip("0")) for x in os.listdir(self.download_path))
        return [x for x in downloaded_ids if x in self.df["id"].values]
    
    def download_data(self) -> None:
        
        already_downloaded_ids = self.get_already_downloaded_ids()

        mask = self.df["id"].isin(already_downloaded_ids)
        remaining = self.df.loc[~mask, "id"]

        n = max(self.download_count - len(already_downloaded_ids), 0)
        sampled_ids = remaining.sample(n=n, random_state=42).tolist() if n > 0 else []

        kepler_ids = already_downloaded_ids + sampled_ids
         
        if n > 0:        
            self.download_all_curves(sampled_ids)
        
        kepler_ids = random.choices(kepler_ids, k=self.sample_count)
        
        for kepler_id in tqdm(kepler_ids, desc="Processing Lightcurves", unit="file"):
            lightcurve = self.get_raw_lightcurve(kepler_id)
            if lightcurve is None:
                continue
            lightcurve_array = self.lightcurve_to_numpy(lightcurve)
            period_distribution = self.get_period_distribution(kepler_id)
            self.save_array(kepler_id, (lightcurve_array, period_distribution))
        
        
            