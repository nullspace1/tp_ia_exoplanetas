import random
import re
from lightkurve.lightcurve import LightCurve
from pandas.core.api import DataFrame as DataFrame
from sympy import li
from StarDataGenerator import StarDataGenerator
import pandas as pd
import numpy as np
from numpy import nan
import batman
import tqdm


class Synthetics(StarDataGenerator):
    def __init__(self, config):
        self.data_path = config["star_catalog_files"]
        self.sample_count = config["synthetic_sample_count"]
        self.planets_per_star = config["planets_per_synthetic_star"]
        StarDataGenerator.__init__(self, config, "synthetic")
        self.download_path = self.download_path + "synthetic/"
        self.distribution_params = config["distribution_params"]
        
    
    def generate_artificial_data(self, lightcurve: pd.DataFrame, planet_radius : float, star_mass: float, star_radius: float,  period: float, period_error: float, first_transit: float) -> pd.DataFrame:
        data = lightcurve.copy()
        data.index.name = "time"
        if "time" not in data.columns:
            data.reset_index(inplace=True)

        period_sim = float(period + np.random.normal(0, period_error))

        G = 6.67408e-11
        M_sun = 1.98892e30
        R_sun = 6.9634e8
        days_to_seconds = 86400
        earth_radius_solar = 0.009168

        rp_sim = planet_radius * earth_radius_solar / star_radius
        a_sim = (G * M_sun * star_mass * (period_sim * days_to_seconds)**2 / (4 * np.pi**2))**(1/3)
        a_sim /= (R_sun * star_radius)
        inc_sim = float(np.random.uniform(85, 90))
        u1_sim, u2_sim = np.random.uniform(0, 0.5, 2)

        t0 = first_transit
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
        lightcurve = super().verify_lightcurve(kepler_id,super().get_raw_lightcurve(kepler_id))
        
        if lightcurve is None:
            return None
        lightcurve_pd = lightcurve.to_pandas()
        planets = self.df[self.df["id"] == kepler_id]
        for _, row in planets.iterrows():
            lightcurve_pd = self.generate_artificial_data(lightcurve_pd, row["planet_radius"],row["star_mass"], row["star_radius"], row["period"],row["period_error"], row["first_transit"])
        return LightCurve(time=lightcurve_pd["time"], flux=lightcurve_pd["flux"])


    
    def sample_period(self, periods: list, star_mass: float, star_radius: float, star_temperature: float) -> float:
        existing = np.array(periods) if len(periods) > 0 else np.array([0])
        alpha = -0.7
        p_min, p_max = 1.0, 300.0
        r = np.random.random()
        period = ((r * (p_max**(alpha + 1) - p_min**(alpha + 1)) + p_min**(alpha + 1)))**(1 / (alpha + 1))
        while np.any(np.isclose(period, existing, rtol=0.02)):
            r = np.random.random()
            period = ((r * (p_max**(alpha + 1) - p_min**(alpha + 1)) + p_min**(alpha + 1)))**(1 / (alpha + 1))
        return float(period)


    def sample_planet_radius(self, star_mass: float, star_radius: float, star_temperature: float, period: float) -> float:
        base_mean = 1.5
        mean = base_mean * (star_radius ** 0.25) * (period / 10) ** 0.1
        sigma = 0.3 * mean
        r = np.random.lognormal(mean=np.log(mean), sigma=np.log(1 + sigma / mean))
        return float(np.clip(r, 0.5, 15.0))
        
    def get_dataset(self, config) -> DataFrame:
        df = pd.read_csv(self.data_path, sep="|", header=0)
        
        df = df[df["nconfp"] == 0]
        df = df[["kepid","logg", "radius","teff"]]
        df.rename(columns={
        "kepid": "id",
        }, inplace=True)
        df["star_mass"] = 10**(df["logg"] - 4.438) * df["radius"]
        df["star_radius"] = df["radius"]
        df["star_temperature"] = df["teff"]
        
        df = df[["id","star_mass", "star_radius", "star_temperature"]]
        
        result_df = pd.DataFrame(columns=["id","star_mass","star_radius", "star_temperature","period","period_error", "planet_radius","first_transit"])

        for idx in tqdm.tqdm(range(len(df)), desc="Adding Synthetic Planets", unit="star"):
            star = df.iloc[idx]
            count = np.random.randint(1, self.planets_per_star)  if self.planets_per_star > 1 else 1
            
            periods = []
            
            for _ in range(count):
                row = star.copy()
                period_days = self.sample_period(periods,row["star_mass"], row["star_radius"], row["star_temperature"])
                period_err_frac = np.random.uniform(0.002, 0.008, count)
                row["planet_radius"] = self.sample_planet_radius(row["star_mass"], row["star_radius"], row["star_temperature"], period_days)
                row["period"] = period_days
                row["period_error"] = np.maximum(1e-6, period_err_frac * period_days)[0]
                row["first_transit"] = np.random.uniform(0, row["period"])
                
                periods.append(period_days)
                
                result_df.loc[len(result_df)] = row
        result_df["id"] = result_df["id"].astype(int)
        return result_df
