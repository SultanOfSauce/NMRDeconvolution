import numpy as np

import nmrMod as nmr

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset


class NMRDataset(Dataset):
    def __init__(self, maxLen = 250000, startSeed = 0, mode = "narrow", spectra = "true"):
        self.maxLen = maxLen
        self.startSeed = startSeed
        self.mode = mode
        self.spectra = spectra
    def __len__(self):
        return self.maxLen

    def __getitem__(self, idx):
        yy, res = nmr.generateRandomSpectrum(idx + self.startSeed, self.mode)
        isPk = np.full_like(yy[self.spectra], False)
        for i in res:
            isPk[i] = True

        #isPk[res[]] = True
        return th.from_numpy(np.float32(yy[self.spectra]).reshape([1,-1])), isPk.reshape([1,-1])