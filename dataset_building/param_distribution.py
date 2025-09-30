from os import close
from matplotlib.dates import TH
import numpy as np
import pandas as pd

THRESHOLD = 20

def get_weight_distribution(weights : pd.DataFrame, value, error) -> pd.DataFrame:
    new_weights = weights.copy()

    if error == 0:
        closest_value = np.argmin(np.abs(new_weights["value"] - value))
        new_weights.loc[closest_value,"weight"] = 1
    else:
        min_value = np.min((new_weights["value"]-value)**2/(error)**2)
        if (min_value > THRESHOLD):
            closest_value = np.argmin(np.abs(new_weights["value"] - value))
            new_weights.loc[closest_value,"weight"] = 1
        else:
            new_weights["weight"] = np.exp(-(new_weights["value"]-value)**2/(error)**2)

    new_weights["weight"] = new_weights["weight"]/np.sum(new_weights["weight"])
    new_weights["weight"] = np.minimum(1, weights["weight"] + new_weights["weight"])
    
    return new_weights
