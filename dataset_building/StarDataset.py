import os
import numpy as np
import pandas as pd
import lightkurve as lk

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
        pass
    
    def get_duration_distribution(self, kepler_id : str) -> np.ndarray:
        pass
    
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
        lightcurve = subset.download_all(flux_column="pdcsap_flux", quality_bitmask="hard")[0]
        
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
    
    def download_data(self) -> None:
        
        for kepler_id in self.df["id"].sample(n=self.sample_count, random_state=42):
            file_path = f"{self.save_path}/{kepler_id}.npz"
            if os.path.exists(file_path):
                continue
            lightcurve = self.download_raw_lightcurve(kepler_id)
            if lightcurve is None:
                continue
            lightcurve_array = self.lightcurve_to_numpy(lightcurve)
            period_distribution = self.get_period_distribution(kepler_id)
            duration_distribution = self.get_duration_distribution(kepler_id)
            self.save_array(f"{kepler_id}", (lightcurve_array, period_distribution, duration_distribution))