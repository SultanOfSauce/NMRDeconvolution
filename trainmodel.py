import numpy as np
import numpy.typing as npt

import NMRAux as nmr
import Layers as ly
from Aux2 import *

from torchinfo import summary

import torch as th
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset,DataLoader

from tqdm import trange

from safetensors.torch import save_model

ML_train = 30000
ML_test  = 1000

batch_size = 64
train_set = NMRDataset(maxLen = ML_train)
test_set = NMRDataset(maxLen = ML_test, startSeed = ML_train)

train_loader: DataLoader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False,
    num_workers=4
)
test_loader: DataLoader = DataLoader(
    dataset=test_set,  batch_size=batch_size, shuffle=False,
    num_workers=4
)



device: th.device = th.device(
    "cuda"
)
print(device)

model: th.nn.Module = NMRSeq().to(device)
print("Initializing model...")
print(summary(model, input_size=(batch_size,1,nmr.nPts)))

optimizer: th.optim.Optimizer = th.optim.Adam(
    params=model.parameters(), lr=0.001, weight_decay=0
)

lossCriterion = nn.CrossEntropyLoss()

EPOCHS = 100

# Loop over epochs
for epoch in trange(EPOCHS, desc="Training epoch"):

    model.train()  # Remember to set the model in training mode before actual training
    # Loop over data
    for batch_idx, batched_datapoint in enumerate(train_loader):
        x, y = batched_datapoint
        x, y = x.to(device), y.to(device)

        # Forward pass + loss computation
        yhat = model(x)
        loss = lossCriterion(yhat, y)

        # Zero-out past gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()



print("Saving model...")
save_model(model, "modelpars2.safetensors")
