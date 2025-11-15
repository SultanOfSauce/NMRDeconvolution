import numpy as np

import nmrMod as nmr
import Layers as ly

import torch as th
import torch.nn as nn
from torch import Tensor
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

def NMRSeq() -> th.nn.Sequential:
    return th.nn.Sequential(

        ly.Inception_variant(1),

        ly.TransposeLayer(-1,-2),


        nn.Linear(
            in_features=136, out_features=64, bias=True
        ),

        nn.ReLU(),

        nn.Linear(
            in_features=64, out_features=32, bias=True
        ),

        nn.ReLU(),

        nn.LSTM(32, 16, bidirectional=True),
        ly.extract_tensor(),

        nn.ReLU(),

        nn.Linear(
            in_features=32, out_features=32, bias=True
        ),
        nn.ReLU(),

        nn.Linear(
            in_features=32, out_features=16, bias=True
        ),
        nn.ReLU(),

        nn.Linear(
            in_features=16, out_features=1, bias=True
        ),

        ly.TransposeLayer(-1,-2),
    )
