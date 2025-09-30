import os
import kagglehub
import json
import pandas as pd
import zipfile
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from requests import get

from param_distribution import get_weight_distribution

config : dict = json.load(open("config.json"))

def download_exoplanet_dataset():
 
    os.environ["KAGGLE_USERNAME"] = config["username"]
    os.environ["KAGGLE_KEY"] = config["key"]
    
    zip_path = kagglehub.dataset_download(
            handle=config["dataset_name"], 
            path="cumulative.csv"
        )
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open("cumulative.csv") as f:
            df = pd.read_csv(f)

    df["period_error"] = (df["koi_period_err1"].abs() + df["koi_period_err2"].abs()) / 2
    df["duration_error"] = (df["koi_duration_err1"].abs() + df["koi_duration_err2"].abs()) / 2
    df["impact_error"] = (df["koi_impact_err1"].abs() + df["koi_impact_err2"].abs()) / 2
    
    df = df[["kepid", "koi_period","koi_duration","koi_impact","period_error", "duration_error", "impact_error", "koi_disposition"]]    
    df.rename(columns={
        "kepid": "id",
        "koi_period": "period",
        "koi_duration": "duration",
        "koi_impact": "impact",
        "koi_disposition": "has_exoplanet"
    }, inplace=True)
    
    df["has_exoplanet"] = df["has_exoplanet"].apply(lambda x: 1 if x == "CONFIRMED" else 0)

    return df.sample(n=config["positive_sample_count"], random_state=42).reset_index(drop=True)

def download_no_exoplanet_dataset():
    
    path = config["secondary_dataset_path"]
    df = pd.read_csv(path, sep="|", header=0)
    
    df = df[df["nconfp"] == 0]
    
    df = df[["kepid"]]
    df.rename(columns={
    "kepid": "id",
    }, inplace=True)
    df["has_exoplanet"] = 0
    df["period"] = np.nan
    df["duration"] = np.nan
    df["impact"] = np.nan
    df["period_error"] = np.nan
    df["duration_error"] = np.nan
    df["impact_error"] = np.nan

    return df.sample(n=config["negative_sample_count"], random_state=42).reset_index(drop=True)

def get_lightcurve(results):
    lc: lk.lightcurve.LightCurve = results.stitch()
    lc = apply_preprocessing(lc)
    lc_table = lc.to_pandas()
    lc_table["time"] = lc.time.value
    lc_table = lc_table.filter(["time", "flux"])
    
    median_time_gap = np.median(np.diff(lc_table["time"]))
    lc_table["idx"] = np.floor((lc_table["time"] - lc_table["time"].iloc[0]) / median_time_gap).astype(int)
    avg_value = lc_table["flux"].mean()
    
    return  lc_table.groupby('idx')['flux'].apply(lambda x: x.mean() if len(x) > 0 else avg_value).to_numpy()


def download_raw_light_curve(kepler_id : str):
    lc = lk.search_lightcurve(f"KIC {kepler_id}").download_all(flux_column="pdcsap_flux")
    return lc

def apply_preprocessing(light_curve : lk.lightcurve.LightCurve):
    normalized : lk.lightcurve.LightCurve = (light_curve.
                                             remove_nans().
                                             remove_outliers().
                                             normalize()
                                             )
    return normalized

def download_processed_light_curve(kepler_id : str):
    lc = lk.search_lightcurve(f"KIC {kepler_id}").download_all(flux_column="pdcsap_flux")
    lc = apply_preprocessing(lc)
    return lc

def save_array(name : str , arr : np.ndarray):
    base_path = f"data/samples/{name}.npy"
    np.save(base_path, arr)
    
def set_up_weights():
    period_weights = pd.DataFrame(columns=["value","weight"])
    period_weights["value"] = np.linspace(config["distribution_params"]["period"]["min"], config["distribution_params"]["period"]["max"], config["distribution_params"]["bins"])
    period_weights["weight"] = np.zeros(config["distribution_params"]["bins"])
            
    duration_weights = pd.DataFrame(columns=["value","weight"])
    duration_weights["value"] = np.linspace(config["distribution_params"]["duration"]["min"], config["distribution_params"]["duration"]["max"], config["distribution_params"]["bins"])
    duration_weights["weight"] = np.zeros(config["distribution_params"]["bins"])
    
    return period_weights, duration_weights

def process_planets_with_exoplanets(exoplanet_df):
    
    for _, row in exoplanet_df.iterrows():
        kepler_id = int(row["id"])
        
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            
            lightcurve_array = get_lightcurve(results)
            os.makedirs("data/samples/positive", exist_ok=True)
            
            period_weights, duration_weights = set_up_weights()
            
            for _, cand in exoplanet_df[exoplanet_df["id"] == kepler_id].iterrows():
                period_weights = get_weight_distribution(period_weights, cand["period"], cand["period_error"])
                duration_weights = get_weight_distribution(duration_weights, cand["duration"], cand["duration_error"])
   

            period_array = period_weights["weight"].to_numpy()
            duration_array = duration_weights["weight"].to_numpy()
            
            save_array("positive/" + str(kepler_id) + "_curve", lightcurve_array)
            save_array("positive/" + str(kepler_id) + "_period", period_array)
            save_array("positive/" + str(kepler_id) + "_duration", duration_array)
            
            print(f"[OK] Processed {kepler_id}")
        else:
            print(f"[WARN] No light curve found for {kepler_id}")


def process_planets_with_no_exoplanets(no_exoplanet_df):
    for _, row in no_exoplanet_df.iterrows():
        kepler_id = int(row["id"])
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            
            lightcurve_array = get_lightcurve(results)
            os.makedirs("data/samples/negative", exist_ok=True)
            
            period_weights, duration_weights = set_up_weights()
            
            
            period_array = period_weights["weight"].to_numpy()
            duration_array = duration_weights["weight"].to_numpy()
            
            save_array("negative/" + str(kepler_id) + "_curve", lightcurve_array)
            save_array("negative/" + str(kepler_id) + "_period", period_array)
            save_array("negative/" + str(kepler_id) + "_duration", duration_array)
            
            print(f"[OK] Processed {kepler_id}")
        else:
            print(f"[WARN] No light curve found for {kepler_id}")

if __name__ == "__main__":
    
    exoplanet_df :  pd.DataFrame = download_exoplanet_dataset()
    no_exoplanet_df : pd.DataFrame = download_no_exoplanet_dataset()

    exoplanet_df.to_csv("data/candidates.csv", index=False)
    no_exoplanet_df.to_csv("data/no_candidates.csv", index=False)
    
    process_planets_with_exoplanets(exoplanet_df)
    process_planets_with_no_exoplanets(no_exoplanet_df)
