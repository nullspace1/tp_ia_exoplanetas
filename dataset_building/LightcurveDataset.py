
from dataclasses import dataclass
from importlib.metadata import distribution
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import tqdm

from typing import Generator
import random

class LightCurveDataset(Dataset):

    
    def __init__(self, pos_path, neg_path, synth_path, prob_pos, prob_neg, prob_synth, lightcurve_length=3000, distribution_length=11):
        super().__init__()
        self.lightcurve_length = lightcurve_length
        self.distribution_length = distribution_length
        self.prob_pos = prob_pos
        self.prob_neg = prob_neg
        self.prob_synth = prob_synth
        self.pos = self._load_folder(pos_path)
        self.neg = self._load_folder(neg_path)
        self.synth = self._load_folder(synth_path)
        self.length = max(len(self.pos), len(self.neg), len(self.synth))


    def _load_folder(self, path):
        data = []
        for f in tqdm.tqdm(os.listdir(path),desc="Loading data"):
            if f.endswith(".npz"):
                d = np.load(f'{path}/{f}', allow_pickle=True)
                if d["arr_0"].shape[0] !=  self.lightcurve_length or d["arr_1"].shape[0] != self.distribution_length:
                    print("Rejected file", f, "with shape", d["arr_0"].shape, "and", d["arr_1"].shape)
                    continue
                data.append((d["arr_0"], d["arr_1"]))
                d.close()
        return data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        source_name = np.random.choice(["pos", "neg", "synth"],
                               p=[self.prob_pos, self.prob_neg, self.prob_synth])
        source = getattr(self, source_name)
        lc, dist = random.choice(source)
        return torch.from_numpy(lc).float(), torch.from_numpy(dist).float()
        
        