import os
import kagglehub
import json
import pandas as pd
import zipfile
import lightkurve as lk
import dataset_building.build_base_signal as rbs
import numpy as np
import matplotlib.pyplot as plt

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
            
    df.loc[:, "mass"] = 10**(df["koi_slogg"] - 4.437) * (df["koi_srad"]**2)   
    df = df[["kepid", "koi_steff", "koi_srad", "ra", "dec", "koi_kepmag","koi_time0bk","koi_impact","koi_prad","koi_period","koi_duration", "koi_disposition"]]
    
    return df.sample(n=config["positive_sample_count"], random_state=42).reset_index(drop=True)

def download_no_exoplanet_dataset():
    
    path = config["secondary_dataset_path"]
    df = pd.read_csv(path, sep="|", header=0)
    
    df = df[df["nconfp"] == 0]
    df = df[["kepid","teff","mass","radius","ra","dec","kmag"]]
    df.loc[:, "koi_time0bk"] = 0
    df.loc[:, "koi_impact"] = 0
    df.loc[:, "koi_period"] = 0
    df.loc[:, "koi_duration"] = 0
    df.loc[:, "koi_disposition"] = "NEGATIVE"
    
    return df.sample(n=config["negative_sample_count"], random_state=42).reset_index(drop=True)

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

def process_planets_with_exoplanets(exoplanet_df):
    for _, row in exoplanet_df.iterrows():
        kepler_id = int(row["kepid"])
        
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            
            lc_df = get_lightcurve_as_df(results)
            save_df(kepler_id, lc_df)

            candidates = exoplanet_df[exoplanet_df["kepid"] == kepler_id]
            bins = np.linspace(-0.5, 0.5, config["model_resolution"]+1)
             
            candidates_dict = {
                "candidate_count": len(candidates),
                "candidates": {}
            }
            os.makedirs("data/light_curves/positive", exist_ok=True)
            modeled_path = f"data/light_curves/positive/{kepler_id}_.json"
            
            count = 1
            for _, cand in candidates.iterrows():
                _, flux_model = rbs.batman_phase_model(
                    cand["koi_srad"],
                    cand["mass"],
                    cand["koi_time0bk"],
                    cand["koi_impact"],
                    cand["koi_prad"],
                    cand["koi_period"],
                    bins
                )

                candidates_dict["candidates"][f"candidate_{count}"] = {
                    "flux": flux_model.tolist(),
                    "period": cand["koi_period"],
                    "duration": cand["koi_duration"]
                }
                
                count += 1
                
            with open(modeled_path, 'w') as f:
                json.dump(candidates_dict, f)
                
            print(f"[OK] Processed {kepler_id}")
        else:
            print(f"[WARN] No light curve found for {kepler_id}")

def save_df(kepler_id, lc_df):
    os.makedirs("data/light_curves/positive", exist_ok=True)
    base_path = f"data/light_curves/positive/{kepler_id}.csv"
    lc_df.to_csv(base_path, index=False)

def get_lightcurve_as_df(results):
    lc: lk.lightcurve.LightCurve = results.stitch()
    lc = apply_preprocessing(lc)
    lc_table = lc.to_pandas()
    lc_table["time"] = lc.time.value
    lc_table = lc_table.filter(["time", "flux"])
    return lc_table

def process_planets_with_no_exoplanets(no_exoplanet_df):
    for _, row in no_exoplanet_df.iterrows():
        kepler_id = int(row["kepid"])
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            
            lc_df = get_lightcurve_as_df(results)
            save_df(kepler_id, lc_df)
            
            modeled_path = f"data/light_curves/negative/{kepler_id}.json"
            dict_to_save = {
                "candidate_count": 0,
            }
            
            with open(modeled_path, 'w') as f:
                json.dump(dict_to_save, f)
            
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
