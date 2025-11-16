import numpy as np

from nmrnet import *

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from safetensors.torch import save_model

from tqdm import trange, tqdm


ML_train = 20000
ML_test  = 500

batch_size = 64
mode = "wide"
train_set = NMRDataset(maxLen = ML_train, spectra = "filtered", mode = mode)
test_set = NMRDataset(maxLen = ML_test, startSeed = ML_train, spectra = "filtered", mode = "wide")

train_loader: DataLoader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False,
    num_workers=4
)
test_loader: DataLoader = DataLoader(
    dataset=test_set,  batch_size=batch_size, shuffle=False,
    num_workers=4
)



device: th.device = th.device(
    "cuda" if th.cuda.is_available() else "cpu"
)
print(device)

model: th.nn.Module = NMRSeq().to(device)
print("Initializing model...")
print(summary(model, input_size=(batch_size,1,nmr.nPts)))

optimizer: th.optim.Optimizer = th.optim.Adam(
    params=model.parameters(), lr=0.001, weight_decay=0
)

lossCriterion = (
#    th.nn.CrossEntropyLoss()
    th.nn.BCEWithLogitsLoss()
)

EPOCHS = 30

eval_losses = np.array([])

# Loop over epochs
for epoch in (bar := trange(EPOCHS, desc="Training   | Training epoch", 
                            bar_format="{desc}:{percentage:3.0f}%|{bar:50}{r_bar}")):

    model.train()  # Remember to set the model in training mode before actual training
    # Loop over data
    for batch_idx, batched_datapoint in enumerate(train_loader):
        
        bar.set_description_str(f"Training   - Batch no {batch_idx:04}/{(ML_train//batch_size + 1):04} | Training epoch")
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
    save_model(model, f"./{mode}/modelpars_{mode}_it_{epoch}.safetensors")
        
    '''
        #Evaluating error:
        model.eval()
        
        
        num_elem: int = 0
        trackingmetric: float = 0
        #trackingcorrect: int = 0
               
        
    with th.no_grad():
        for o, batched_datapoint_e in enumerate(train_loader):
            
            bar.set_description_str(f"Evaluating - Batch no {o:04}/{(ML_train//batch_size + 1):04} | Training epoch")
                
            x_e, y_e = batched_datapoint_e
            x_e, y_e = x_e.to(device), y_e.to(device)
            modeltarget_e = model(x_e)
                
            trackingmetric += sumCriterion(modeltarget_e, y_e).item()
                #trackingcorrect += ypred_e.eq(y_e.view_as(ypred_e)).sum().item()
                
            num_elem += x_e.shape[0]
        eval_losses = np.append(eval_losses, trackingmetric / num_elem)
    '''

print("Saving model...")
save_model(model, f"./{mode}/modelpars_{mode}_final.safetensors")
