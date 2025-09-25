import os
import kagglehub
import json
import pandas as pd
import zipfile
import lightkurve as lk
import recover_base_signal as rbs
import numpy as np

config : dict = json.load(open("config.json"))

def download_main_dataset():

    
    os.environ["KAGGLE_USERNAME"] = config["username"]
    os.environ["KAGGLE_KEY"] = config["key"]
    
    zip_path = kagglehub.dataset_download(
            handle=config["dataset_name"], 
            path="cumulative.csv"
        )
    
    ## POR QUE ES UN ZIP HIJO DE PUTA
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open("cumulative.csv") as f:
            df = pd.read_csv(f)
    
    return df

def download_secondary_dataset():
    
    path = config["secondary_dataset_path"]
    df = pd.read_csv(path, sep="|", header=0)
    
    return df

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

if __name__ == "__main__":
    
    TOTAL_ENTRIES = 450
    
    exoplanet_df :  pd.DataFrame = download_main_dataset()
    no_exoplanet_df : pd.DataFrame = download_secondary_dataset()
    
    
    no_exoplanet_df = no_exoplanet_df[no_exoplanet_df["nconfp"] == 0]
    
    exoplanet_lookup_table = exoplanet_df[["kepid", "koi_steff", "koi_srad", "ra", "dec", "koi_kepmag","koi_time0bk","koi_impact","koi_prad","koi_period","koi_duration", "koi_disposition"]]
    exoplanet_lookup_table.to_csv("data/candidates_lookup_table.csv", index=False)
    
    no_exoplanet_lookup_table = no_exoplanet_df[["kepid","teff","radius","ra","dec","kmag"]]
    
    no_exoplanet_lookup_table.loc[:, "koi_time0bk"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_impact"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_period"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_duration"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_disposition"] = "NEGATIVE"
    
    exoplanet_lookup_table = exoplanet_lookup_table.sample(n=TOTAL_ENTRIES, random_state=42)
    no_exoplanet_lookup_table = no_exoplanet_lookup_table.sample(n=TOTAL_ENTRIES, random_state=42)
    
    no_exoplanet_lookup_table.to_csv("data/no_candidates_lookup_table.csv", index=False)
    
    for _, row in exoplanet_df.iterrows():
        kepler_id = int(row["kepid"])
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            lc: lk.lightcurve.LightCurve = results.stitch()
            lc = apply_preprocessing(lc)
            lc_table = lc.to_pandas()
            lc_table["time"] = lc.time.value
            lc_table = lc_table.filter(["time", "flux"])

            os.makedirs("data/light_curves/positive/base", exist_ok=True)
            base_path = f"data/light_curves/positive/base/{kepler_id}.csv"
            lc_table.to_csv(base_path, index=False)

            candidates = exoplanet_df[exoplanet_df["kepid"] == kepler_id]
            combined_flux = np.ones_like(lc_table["time"].to_numpy(), dtype=float)

            for _, cand in candidates.iterrows():
                time, flux_model = rbs.build_base_signal(
                    lc_table,
                    cand["koi_srad"],
                    cand["koi_time0bk"],
                    cand["koi_impact"],
                    cand["koi_prad"],
                    cand["koi_period"],
                    cand["koi_duration"],
                )
                combined_flux *= flux_model

            new_signal = lc_table.copy()
            new_signal["flux"] = combined_flux
            os.makedirs("data/light_curves/positive/modeled", exist_ok=True)
            modeled_path = f"data/light_curves/positive/modeled/{kepler_id}.csv"
            new_signal.to_csv(modeled_path, index=False)

            print(f"[OK] Processed {kepler_id} → {base_path}, {modeled_path}")
        else:
            print(f"[WARN] No light curve found for {kepler_id}")
            
    for _, row in no_exoplanet_df.iterrows():
        kepler_id = int(row["kepid"])
        try:
            results = download_raw_light_curve(kepler_id)
        except Exception as e:
            print(f"[WARN] Failed to download light curve for {kepler_id}: {e}")
            continue

        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            lc: lk.lightcurve.LightCurve = results.stitch()
            lc = apply_preprocessing(lc)
            lc_table = lc.to_pandas()
            lc_table["time"] = lc.time.value
            lc_table = lc_table.filter(["time", "flux"])

            os.makedirs("data/light_curves/negative/base", exist_ok=True)
            base_path = f"data/light_curves/negative/base/{kepler_id}.csv"
            lc_table.to_csv(base_path, index=False)
            
            new_signal = lc_table.copy()
            new_signal["flux"] = np.ones_like(new_signal["flux"], dtype=float)
            os.makedirs("data/light_curves/negative/modeled", exist_ok=True)
            modeled_path = f"data/light_curves/negative/modeled/{kepler_id}.csv"
            new_signal.to_csv(modeled_path, index=False)

            print(f"[OK] Processed {kepler_id} → {base_path}")
        else:
            print(f"[WARN] No light curve found for {kepler_id}")
