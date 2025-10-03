from lightkurve.lightcurve import LightCurve
from pandas.core.api import DataFrame as DataFrame
from StarDataset import StarDataset
import pandas as pd
import numpy as np
from numpy import nan
import batman


class Synthetics(StarDataset):
    def __init__(self, config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["synthetic_sample_count"]
        StarDataset.__init__(self, config, "synthetic")
        
    
    def generate_artificial_data(self,lightcurve: pd.DataFrame, period: float,period_error: float,duration: float,duration_error: float) -> pd.DataFrame:
        
        data = lightcurve.copy()
        data.reset_index(inplace=True)

        period_sim = float(period + np.random.normal(0, period_error))
        duration_sim = float(duration + np.random.normal(0, duration_error))

        rp_sim = float(np.random.uniform(0.05, 0.15))
        a_sim = float(np.random.uniform(10, 25))
        inc_sim = float(np.random.uniform(85, 90))
        u1_sim, u2_sim = float(np.random.uniform(0, 0.5)), float(np.random.uniform(0, 0.5))

        t0 = float(data["time"].iloc[0] + period_sim / 2)

        params = batman.TransitParams()
        params.t0 = t0
        params.per = period_sim
        params.rp = rp_sim
        params.a = a_sim
        params.inc = inc_sim
        params.ecc = 0.0
        params.w = 90.0
        params.u = [u1_sim, u2_sim]
        params.limb_dark = "quadratic"

        time_array = data["time"].values.astype(float)

        m = batman.TransitModel(params, time_array)
        transit_flux = m.light_curve(params)

        transit_flux = 1 - ((1 - transit_flux) * (duration_sim / duration))

        data["flux"] = data["flux"] * transit_flux

        return data
        
    def download_raw_lightcurve(self,kepler_id : str):
        lightcurve = super().download_raw_lightcurve(kepler_id)
        lightcurve_pd = lightcurve.to_pandas()
        planets = self.df[self.df["id"] == kepler_id]
        for _, row in planets.iterrows():
            lightcurve_pd = self.generate_artificial_data(lightcurve_pd, row["period"],row["period_error"],row["duration"],row["duration_error"])
        return LightCurve(time=lightcurve_pd["time"], flux=lightcurve_pd["flux"])

        
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
        
        # Configuration bounds
        period_min = float(config["distribution_params"]["period"]["min"])
        period_max = float(config["distribution_params"]["period"]["max"])
        duration_min = float(config["distribution_params"]["duration"]["min"])
        duration_max = float(config["distribution_params"]["duration"]["max"])
        
        for _ in range(self.sample_count // 4):
            count = np.random.randint(1, 10) 
            new_exoplanets = df.sample(n=count, random_state=42).reset_index(drop=True)
            new_exoplanets["has_exoplanet"] = 1
            
            # Period (days): uniform within configured bounds
            periods_days = np.random.uniform(period_min, period_max, count)
            new_exoplanets["period"] = periods_days
            
            duty_cycle = np.random.uniform(0.005, 0.05, count)
            durations_hours = duty_cycle * periods_days * 24.0
            durations_hours = np.clip(durations_hours, duration_min, duration_max)
            new_exoplanets["duration"] = durations_hours

            # Errors as percentage of values (non-zero)
            period_err_frac = np.random.uniform(0.02, 0.08, count)
            new_exoplanets["period_error"] = np.maximum(1e-6, period_err_frac * periods_days)

            duration_err_frac = np.random.uniform(0.05, 0.15, count)
            new_exoplanets["duration_error"] = np.maximum(1e-6, duration_err_frac * durations_hours)
                
            df = pd.concat([df, new_exoplanets])

        df = df[df["has_exoplanet"] == 1]
        df = df[["id", "period", "duration", "period_error", "duration_error"]]
        
        return df
