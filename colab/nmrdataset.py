import numpy as np

import nmrgen as nmr

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset


class NMRDataset(Dataset):
    def __init__(self, maxLen = 250000, startSeed = 0, mode = "narrow", spectra = "true", withPeaks = False):
        self.maxLen = maxLen
        self.startSeed = startSeed
        self.mode = mode
        self.spectra = spectra
        self.withPeaks = withPeaks
        
    def __len__(self):
        return self.maxLen

    def __getitem__(self, idx):
        yy, res = nmr.generateRandomSpectrum(idx + self.startSeed, self.mode)
        
        resInd, resWidth = res
        
        isPk = np.full_like(yy[self.spectra], False)
        for i in resInd:
            isPk[i] = True

        if self.withPeaks:
            return th.from_numpy(np.float32(yy[self.spectra])), numpy.concatenate(isPk.reshape([1,-1]), resWidth)
        else:
            return th.from_numpy(np.float32(yy[self.spectra])), th.from_numpy(isPk.reshape([1,-1]))