import numpy as np

from nmrnet import *
from nmrdataset import *

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_model
from torchinfo import summary

from tqdm import trange

mode = 'wide'

ML = 10000
ML_test = 1000
batch_size = 32
workers = 2

train_set = NMRDataset(maxLen = ML     , mode = mode)
test_set  = NMRDataset(maxLen = ML_test, mode = mode, startSeed = ML)
train_loader: DataLoader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False,
    num_workers=workers
)
test_loader: DataLoader = DataLoader(
    dataset=test_set,  batch_size=batch_size, shuffle=False,
    num_workers=workers
)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = NMRNet().to(device)
summary(model, input_size=(batch_size, nmr.nPts))

criterion = nn.BCEWithLogitsLoss()

# 2. Define Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# 3. Define Number of Epochs
NUM_EPOCHS = 100

# 4. Arrays to store loss history
train_losses = []
test_losses = []


for epoch in (bar := trange(NUM_EPOCHS, desc="Training   | Training epoch", 
                            bar_format="{desc}:{percentage:3.0f}%|{bar:50}{r_bar}")):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    
    for i, nos in enumerate(train_loader):
        inputs, targets = nos
        bar.set_description_str(f"Training   - Batch no {i:04}/{(ML//batch_size + 1):04} | Training epoch")
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # --- Forward Pass ---
        # The model now returns raw logits
        outputs = model(inputs)
        
        # --- Calculate Loss ---
        # criterion compares the raw logits (outputs) with the targets
        loss = criterion(outputs, targets)
        
        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * inputs.size(0)
        
    save_model(model, f"./{mode}/modelpars_{mode}_i_{epoch}.safetensors")
        
        

    # Calculate average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # --- Evaluation (Test) Phase ---
    model.eval()
    running_test_loss = 0.0
    
    # Use torch.no_grad() to disable gradient calculations
    with th.no_grad():
        for i, nos in enumerate(test_loader):
            inputs, targets = nos
            bar.set_description_str(f"Validating - Batch no {i:04}/{(ML_test//batch_size + 1):04} | Training epoch")
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            running_test_loss += loss.item() * inputs.size(0)

    # Calculate average test loss for the epoch
    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)

print("Saving model...")
save_model(model, f"./{mode}/modelpars_{mode}_final.safetensors")
np.savetxt(f'./{mode}/trainlosses.csv', np.array(train_losses), delimiter = ',')
np.savetxt(f'./{mode}/testlosses.csv' , np.array(test_losses) , delimiter = ',')
