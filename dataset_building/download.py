import os
import kagglehub
import json
import pandas as pd
import zipfile
import lightkurve as lk

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
    
    exoplanet_df :  pd.DataFrame = download_main_dataset()
    no_exoplanet_df : pd.DataFrame = download_secondary_dataset()
    
    no_exoplanet_df = no_exoplanet_df[no_exoplanet_df["nconfp"] == 0]
    
    exoplanet_lookup_table = exoplanet_df[["kepid", "koi_steff", "koi_srad", "ra", "dec", "koi_kepmag","koi_time0bk","koi_impact","koi_period","koi_duration", "koi_disposition"]]
    exoplanet_lookup_table.to_csv("data/candidates_lookup_table.csv", index=False)
    
    
    no_exoplanet_lookup_table = no_exoplanet_df[["kepid","teff","radius","ra","dec","kmag"]]
    
    no_exoplanet_lookup_table.loc[:, "koi_time0bk"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_impact"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_period"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_duration"] = 0
    no_exoplanet_lookup_table.loc[:, "koi_disposition"] = "NEGATIVE"
    
    no_exoplanet_lookup_table.to_csv("data/no_candidates_lookup_table.csv", index=False)
    
    for kepler_id in exoplanet_df["kepid"].unique():
        results = download_raw_light_curve(kepler_id)
        if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
            lc : lk.lightcurve.LightCurve = results.stitch()
            lc   = apply_preprocessing(lc)
            lc_table = lc.to_pandas()
            lc_table["time"] = lc.time.value
            lc_table = lc_table.filter(['time','flux','flux_err'])
            lc_table.to_csv(f"data/light_curves/{kepler_id}.csv", index=False)
            
