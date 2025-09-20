import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_gaps(df : pd.DataFrame):
    time_gaps = df["time"].dropna().diff().dropna()
    median_time_gap : float  = np.median(time_gaps) if time_gaps.size > 0 else np.nan
    df["time_gap"] = (df["time"].diff() - median_time_gap) > median_time_gap *  50
    return df

data = pd.read_csv("data/10848459.csv")
data = fill_gaps(data)

plt.scatter(data["time"], data["flux"])
plt.show()
plt.scatter(data["time"], data["time_gap"], color="red")
plt.show()

