import random
import re
from lightkurve.lightcurve import LightCurve
from pandas.core.api import DataFrame as DataFrame
from StarDataset import StarDataset
import pandas as pd
import numpy as np
from numpy import nan
import batman
import tqdm


class Synthetics(StarDataset):
    def __init__(self, config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["synthetic_sample_count"]
        self.planets_per_star = config["planets_per_synthetic_star"]
        StarDataset.__init__(self, config, "synthetic")
        self.download_path = self.download_path + "synthetic/"
        
    
    def generate_artificial_data(self, lightcurve: pd.DataFrame, star_mass: float, star_radius: float, period: float, period_error: float) -> pd.DataFrame:
        data = lightcurve.copy()
        data.index.name = "time"
        if "time" not in data.columns:
            data.reset_index(inplace=True)

        period_sim = float(period + np.random.normal(0, period_error))

        G = 6.67408e-11
        M_sun = 1.98892e30
        R_sun = 6.9634e8
        days_to_seconds = 86400
        earth_radius = 0.009168

        rp_sim = float(np.random.uniform(1, 15)) * earth_radius / star_radius
        a_sim = (G * M_sun * star_mass * (period_sim * days_to_seconds)**2 / (4 * np.pi**2))**(1/3)
        a_sim /= (R_sun * star_radius)
        inc_sim = float(np.random.uniform(85, 90))
        u1_sim, u2_sim = np.random.uniform(0, 0.5, 2)

        t0 = float(data["time"].iloc[0] + np.random.uniform(0, 1) * period_sim / 2)

        params = batman.TransitParams()
        params.t0 = t0
        params.per = period_sim
        params.rp = rp_sim
        params.a = a_sim
        params.inc = inc_sim
        params.ecc = np.random.uniform(0, 0.05)
        params.w = 90.0
        params.u = [u1_sim, u2_sim]
        params.limb_dark = "quadratic"

        time_array = data["time"].values.astype(float)
        m = batman.TransitModel(params, time_array)
        transit_flux = m.light_curve(params)

        data["flux"] = data["flux"] * transit_flux
        return data

        
    def get_raw_lightcurve(self,kepler_id : str):
        lightcurve = super().get_raw_lightcurve(kepler_id)
        if lightcurve is None:
            return None
        lightcurve_pd = lightcurve.to_pandas()
        planets = self.df[self.df["id"] == kepler_id]
        for _, row in planets.iterrows():
            lightcurve_pd = self.generate_artificial_data(lightcurve_pd,row["star_mass"], row["star_radius"], row["period"],row["period_error"])
        return LightCurve(time=lightcurve_pd["time"], flux=lightcurve_pd["flux"])

        
    def get_dataset(self, config) -> DataFrame:
        df = pd.read_csv(self.data_path, sep="|", header=0)
        
        df = df[df["nconfp"] == 0]
        df = df[["kepid","logg", "radius"]]
        df.rename(columns={
        "kepid": "id",
        }, inplace=True)
        df["has_exoplanet"] = nan
        df["period"] = nan
        df["period_error"] = nan
        df["star_mass"] = 10**(df["logg"] - 4.438) * df["radius"]
        df["star_radius"] = df["radius"]
        
        df = df[["id","star_mass", "star_radius"]]
        
        result_df = pd.DataFrame(columns=["id","star_mass","star_radius", "period", "period_error"])
        
        period_min = float(config["distribution_params"]["period"]["min"])
        period_max = float(config["distribution_params"]["period"]["max"])

        
        for idx in tqdm.tqdm(range(len(df)), desc="Adding Synthetic Planets", unit="star"):
            star = df.iloc[idx]
            count = np.random.randint(1, self.planets_per_star)  if self.planets_per_star > 1 else 1
            
            for _ in range(count):
                row = star.copy()
                period_days = np.random.uniform(period_min, period_max)
                period_err_frac = np.random.uniform(0.002, 0.008, count)
                row["period"] = period_days
                row["period_error"] = np.maximum(1e-6, period_err_frac * period_days)[0]
                
                result_df.loc[len(result_df)] = row
        result_df["id"] = result_df["id"].astype(int)
        return result_df
