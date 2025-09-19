import os
import kagglehub
import json
import csv
import pandas as pd
import zipfile
import lightkurve as lk
import matplotlib.pyplot as plt



def download_main_dataset():
    config : dict = json.load(open("config.json"))
    
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

def download_light_curves(kepler_id : str):
    lc = lk.search_lightcurve(f"KIC {kepler_id}").download_all(flux_column="pdcsap_flux")
    return lc

def process_light_curve(light_curve : lk.lightcurve.LightCurve):
    normalized : lk.lightcurve.LightCurve = (light_curve.remove_nans().remove_outliers().normalize())
    return normalized
    
df :  pd.DataFrame = download_main_dataset()

for kepler_id in df["kepid"]:
    results = download_light_curves(kepler_id)
    if results is not None and isinstance(results, lk.LightCurveCollection) and len(results) > 0:
        lc : lk.lightcurve.LightCurve = results.stitch()
        lc   = process_light_curve(lc)
        lc_table = lc.to_pandas()
        lc_table["time"] = lc.time.value
        lc_table = lc_table.filter(['time','pdcsap_flux','pdcsap_flux_err'])
        lc_table.to_csv(f"data/{kepler_id}.csv", index=False)
                