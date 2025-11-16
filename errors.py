import numpy as np
import matplotlib.pyplot as plt

%reload_ext autoreload
%autoreload 2

from nmrnet import *
import nmrMod as nmr

import torch as th
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from safetensors.torch import load_model
from torchinfo import summary

#from collections.abc import Callable
from tqdm import trange, tqdm


ML = 20000
ML_test = 500
batch_size = 64
train_set = NMRDataset(maxLen = ML, mode = "wide")
test_set = NMRDataset(maxLen = ML_test, startSeed = ML)

train_loader: DataLoader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False,
    num_workers=4
)
test_loader: DataLoader = DataLoader(
    dataset=test_set,  batch_size=batch_size, shuffle=False,
    num_workers=4
)


epochs = 100
lossCriterion = nn.CrossEntropyLoss(reduction = 'mean')

errArr = np.zeros(epochs)
testErrArr = np.zeros(epochs)
bar = trange(epochs)

for i in bar:
    modeltmp = NMRSeq().to(device)
    load_model(model, f"./cross100/modelpars_narrow_it_{i}.safetensors")
    modeltmp.eval()
    
    err = 0.
    testErr = 0.
    
    with th.no_grad():
        #Train
        for i, batched_datapoint_e in enumerate(train_loader):
            bar.set_description_str(desc = f"train, {i}")
            x_e, y_e = batched_datapoint_e
            x_e, y_e = x_e.to(device), y_e.to(device)
            modeltarget_e = modeltmp(x_e)
            err += lossCriterion(modeltarget_e, y_e).item()
            
        #Test
        for i, batched_datapoint_e in enumerate(test_loader):
            bar.set_description_str(desc = f"test, {i}")            
            x_e, y_e = batched_datapoint_e
            x_e, y_e = x_e.to(device), y_e.to(device)
            modeltarget_e = modeltmp(x_e)
            testErr += lossCriterion(modeltarget_e, y_e)
        
        errArr[i] = err/ML
        testErrArr[i] = testErr/ML_test
    
    
    del modeltmp

np.savetxt("./err.csv", errArr, delimiter=',')
np.savetxt("./testerr.csv", testErrArr, delimiter=',')