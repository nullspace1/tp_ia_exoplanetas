import os
import json
from turtle import pos
import pandas as pd
import numpy as np
from math import nan
from Positives import Positives
from Negatives import Negatives
from Synthetics import Synthetics

def main():
    
    config = json.load(open("config.json"))
    
    positives_dataset = Positives(config)
    negatives_dataset = Negatives(config)
    synthetics_dataset = Synthetics(config)

    synthetics_dataset.download_data()
    negatives_dataset.download_data()
    positives_dataset.download_data()
    
if __name__ == "__main__":
    main()
