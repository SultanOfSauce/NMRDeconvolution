from torch import cat
import torch.nn as nn
from torch.nn.functional import relu

def Inception_piece(in_channels, out_channels, kernel_size):
    padding = kernel_size - 1
    return nn.Sequential(
                nn.Conv1d(in_channels, 1           , kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
            )

class Inception(nn.Module):
    """
    'Inception' Layer... Pretty much parallel CNNs.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        cnn_configs = [
            (16, 2),
            (32, 4),
            (64, 8),
            (16, 32),
            (8, 64),
        ]
        
        self.branches = nn.ModuleList()
        for out_c, k_s in cnn_configs:
            # Inception
            branch = Inception_piece(in_channels, out_c, k_s)
            self.branches.append(branch)
        
        
        self.total_out_channels = sum(out_c for out_c, _ in cnn_configs) # 136

    def forward(self, x):
        # x shape: (B, C_in=1, L)        
        branch_outputs = [branch(x) for branch in self.branches]
        
        #(B, 136, L)
        return cat(branch_outputs, dim=1)


class NMRNet(nn.Module):
    def __init__(self, input_sequence_length=None):
        super().__init__()

        lstm_hidden_size = 16
        lstm_input_size = 32

        # Inception Block
        self.parallel_cnn_block = Inception(in_channels=1)
        cnn_out_channels = self.parallel_cnn_block.total_out_channels # 136
        
        # Tdd can be done with a big enough conv1d
        
        self.tdd1 = nn.Conv1d(cnn_out_channels, 64, kernel_size=1)
        self.tdd2 = nn.Conv1d(64, lstm_input_size, kernel_size=1) # 32 channels

        # LSTM
        # Input: (B, L, 32). Output: (B, L, 2 * 16) = (B, L, 32)
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_channels = lstm_hidden_size * 2 # 32

        self.tdd3 = nn.Conv1d(lstm_out_channels, 32, kernel_size=1)        
        self.tdd4 = nn.Conv1d(32, 16, kernel_size=1)
        self.tdd_final = nn.Conv1d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # The input x is assumed to be (B, L)
        
        # Reshape: (B, L) -> (B, C=1, L)
        x = x.unsqueeze(1) 
        
        # Output: (B, 136, L)
        x = relu(self.parallel_cnn_block(x))        
        # TDD 1: (B, 136, L) -> (B, 64, L)
        x = relu(self.tdd1(x))        
        # TDD 2: (B, 64, L) -> (B, 32, L)
        x = relu(self.tdd2(x))

        
        # Permute for LSTM: (B, C, L) -> (B, L, C)
        # x is now (B, L, 32)
        x = x.permute(0, 2, 1) 
        
        # Output: (B, L, 2 * 32) = (B, L, 64)
        x, _ = self.bilstm(x)
        
        # Permute back for TDD: (B, L, C) -> (B, C, L)
        # x is now (B, 64, L)
        x = x.permute(0, 2, 1) 

        # TDD 3: (B, 64, L) -> (B, 32, L)
        x = relu(self.tdd3(x))        
        # TDD 4: (B, 32, L) -> (B, 16, L)
        x = relu(self.tdd4(x))
        # TDD 5 (Final): (B, 16, L) -> (B, 1, L)
        x = self.tdd_final(x)
        
        # Output the final logits, and not a sigmoid
        final_output = x
        
        return final_output