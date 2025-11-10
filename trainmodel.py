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

ML_train = 10000
ML_test  = 500

batch_size = 512
train_set = NMRDataset(maxLen = ML_train)
test_set = NMRDataset(maxLen = ML_test, startSeed = ML_train)

train_loader: DataLoader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False,
    num_workers=2
)
test_loader: DataLoader = DataLoader(
    dataset=test_set,  batch_size=batch_size, shuffle=False,
    num_workers=2
)



device: th.device = th.device(
    "cuda" if th.cuda.is_available() else "cpu"
)

model: th.nn.Module = NMRSeq().to(device)
print("Initializing model...")
print(summary(model, input_size=(batch_size,1,nmr.nPts)))

optimizer: th.optim.Optimizer = th.optim.Adam(
    params=model.parameters(), lr=0.001, weight_decay=0
)

lossCriterion = nn.CrossEntropyLoss()

EPOCHS = 100
BATCH_SIZE = 32

eval_losses = []
eval_acc = []
test_acc = []

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

    model.eval()  # Remember to set the model in evaluation mode before evaluating it

    num_elem: int = 0
    trackingmetric: float = 0
    trackingcorrect: int = 0

    # Since we are just evaluating the model, we don't need to compute gradients
    with th.no_grad():
        # ... by looping over training data again
        for _, batched_datapoint_e in enumerate(train_loader):
            x_e, y_e = batched_datapoint_e
            x_e, y_e = x_e.to(device), y_e.to(device)
            modeltarget_e = model(x_e)
            ypred_e = th.argmax(modeltarget_e, dim=1, keepdim=True)
            trackingmetric += lossCriterion(modeltarget_e, y_e).item()
            trackingcorrect += ypred_e.eq(y_e.view_as(ypred_e)).sum().item()
            num_elem += x_e.shape[0]
        eval_losses.append(trackingmetric / num_elem)
        eval_acc.append(trackingcorrect / num_elem)

print("Saving model...")
save_model(model, "modelpars2.safetensors")
np.array(eval_losses).tofile('eval_losses.csv', sep = ',')
np.array(eval_acc).tofile('eval_acc.csv', sep = ',')
