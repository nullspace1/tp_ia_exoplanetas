import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
import shutil

class StarDataset:
    
    df : pd.DataFrame
    sample_count: int
    lightcurve_length: int
    min_lightcurve_length: int
    save_path: str
    period: dict
    duration: dict
    
    def __init__(self, config, path_key):
        self.df = self.get_dataset(config)
        self.lightcurve_length = config["lightcurve_length"]
        self.min_lightcurve_length = config["min_lightcurve_length"]
        self.save_path = config[path_key]
        self.cache_path = config["cache_path"]
        self.period = {
            "min": config["distribution_params"]["period"]["min"],
            "max": config["distribution_params"]["period"]["max"],
            "bins": config["distribution_params"]["period"]["bins"]
        }
        self.duration = {
            "min": config["distribution_params"]["duration"]["min"],
            "max": config["distribution_params"]["duration"]["max"],
            "bins": config["distribution_params"]["duration"]["bins"]
        }
        
    def get_dataset(self, config) -> pd.DataFrame:
        pass
    
    def get_period_distribution(self, kepler_id : str) -> np.ndarray:
        return self.get_distribution(kepler_id, self.period, "period")
    
    def get_duration_distribution(self, kepler_id : str) -> np.ndarray:
        return self.get_distribution(kepler_id, self.duration, "duration")
    
    def get_distribution(self, kepler_id : str, variable : dict, variable_name : str) -> np.ndarray:
        planets = self.df[self.df["id"] == kepler_id]
        
        weights = np.zeros(variable["bins"])
        values = np.linspace(variable["min"], variable["max"], variable["bins"]) 
        
        for _, row in planets.iterrows():
            variable_value = row[variable_name]
            variable_error = row[f"{variable_name}_error"]
            
            weights_to_add = np.zeros(variable["bins"])
            
            min_value = np.min((values-variable_value)**2/(variable_error)**2)
            if ((min_value > 100) == np.True_ or variable_error == 0):
                close_value = np.min(np.abs(values - variable_value))
                weights_to_add = np.exp(-(values-variable_value)**2/(close_value)**2)
            else:
                weights_to_add = np.exp(-(values-variable_value)**2/(variable_error)**2)

            weights_to_add = weights_to_add/np.sum(weights_to_add)
            weights = np.minimum(1, weights + weights_to_add)
        
        return weights
    
    def download_raw_lightcurve(self, kepler_id : str) -> lk.lightcurve.LightCurve | None:
        search = lk.search_lightcurve(
        f"KIC {kepler_id}",
        mission="Kepler",
        cadence="long",
        exptime=1800
        )
        if len(search) == 0:
            return None
        segment = np.random.randint(0, len(search))
        subset = search[segment: segment + 1]
        try:
            lightcurve = subset.download_all(flux_column="pdcsap_flux", quality_bitmask="hard")[0]
        except:
            return None
        
        if (isinstance(lightcurve, lk.lightcurve.LightCurve) and lightcurve.time.size > self.min_lightcurve_length):
            return lightcurve.remove_nans().remove_outliers().normalize()
        else:
            return None
    
    
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
        base_path = f"{self.save_path}/{kepler_id}.npz"
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
            
    
    def download_data(self) -> None:
        
        kepler_ids = self.df["id"].sample(n=self.sample_count, random_state=42).tolist()
        
        kepler_ids_to_process = []
        for kepler_id in kepler_ids:
            file_path = f"{self.save_path}/{kepler_id}.npz"
            if not os.path.exists(file_path):
                kepler_ids_to_process.append(kepler_id)
        
        print(f"Processing {len(kepler_ids_to_process)} lightcurves...")
        
        download_count = 0
        successful_downloads = 0
        
        for kepler_id in tqdm(kepler_ids_to_process, desc="Downloading lightcurves", unit="file"):
            lightcurve = self.download_raw_lightcurve(kepler_id)
            if lightcurve is None:
                continue
            lightcurve_array = self.lightcurve_to_numpy(lightcurve)
            period_distribution = self.get_period_distribution(kepler_id)
            duration_distribution = self.get_duration_distribution(kepler_id)
            self.save_array(f"{kepler_id}", (lightcurve_array, period_distribution, duration_distribution))
            successful_downloads += 1
            download_count += 1
            if download_count % 20 == 0:
                self.clean_cache()
                download_count = 0
        
        print(f"Successfully downloaded {successful_downloads} lightcurves.")
            