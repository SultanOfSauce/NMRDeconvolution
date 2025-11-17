from torch import cat
import torch.nn as nn
from torch.nn.functional import relu

# A helper function to create a Conv1d layer with 'same' padding
# to ensure the output sequence length matches the input sequence length.
def create_same_padded_conv1d(in_channels, out_channels, kernel_size):
    """Creates a Conv1d layer that maintains sequence length."""
    padding = kernel_size // 2
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

def Inception_piece(in_channels, out_channels, kernel_size):
    padding = kernel_size - 1
    return nn.Sequential(
                nn.Conv1d(in_channels, 1           , kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
            )

class ParallelCNNBlock(nn.Module):
    """
    Implements the 5 nested 1D Convolutional Neural Network branches.
    Each branch is applied to the input, and their outputs are concatenated.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Define the 5 nested CNN branches (Output Channels, Kernel Size)
        # This uses the corrected (channels, kernel_size) mapping.
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
    """
    A sequential model combining parallel CNNs, Time-Distributed Dense (TDD) layers,
    and a Bidirectional LSTM for sequence-to-sequence prediction.
    """
    def __init__(self, input_sequence_length=None):
        super().__init__()

        lstm_hidden_size = 16
        lstm_input_size = 32 # This is the channel size before permuting for LSTM

        # 1. Parallel CNN Block
        self.parallel_cnn_block = ParallelCNNBlock(in_channels=1)
        cnn_out_channels = self.parallel_cnn_block.total_out_channels # 136
        
        # Tdd can be done with conv1d, big enough
        
        # 2. Time Distributed Dense Layer 1 (TDD 1)
        self.tdd1 = nn.Conv1d(cnn_out_channels, 64, kernel_size=1)
        
        # 3. Time Distributed Dense Layer 2 (TDD 2)
        self.tdd2 = nn.Conv1d(64, lstm_input_size, kernel_size=1) # -> 32 channels

        # 4. Bidirectional LSTM
        # Input: (B, L, 32). Output: (B, L, 2 * 32) = (B, L, 64)
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,  # Important: input/output format is (Batch, Length, Channels)
            bidirectional=True
        )
        lstm_out_channels = lstm_hidden_size * 2 # 32

        # 5. Time Distributed Dense Layer 3 (TDD 3)
        self.tdd3 = nn.Conv1d(lstm_out_channels, 32, kernel_size=1)
        
        # 6. Time Distributed Dense Layer 4 (TDD 4)
        self.tdd4 = nn.Conv1d(32, 16, kernel_size=1)

        # 7. Time Distributed Dense Layer 5 (TDD 5) - Final Output
        # The output length is the same as the input vector length (L).
        self.tdd_final = nn.Conv1d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # The input 'x' is assumed to be (Batch_Size, Sequence_Length), e.g., (B, L)
        
        # 1. Reshape for Conv1d: (B, L) -> (B, C=1, L)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) 
        
        # --- CNN Blocks (Feature Extraction) ---
        # Output: (B, 136, L)
        x = relu(self.parallel_cnn_block(x))
        
        # --- Time Distributed Dense Layers before LSTM ---
        
        # TDD 1: (B, 126, L) -> (B, 64, L)
        x = relu(self.tdd1(x))
        
        # TDD 2: (B, 64, L) -> (B, 32, L)
        x = relu(self.tdd2(x))

        # ----------------------------------------------------
        # The LSTM requires input in the format (Batch, Length, Channels)
        
        # 2. Permute for LSTM: (B, C, L) -> (B, L, C)
        # x is now (B, L, 32)
        x = x.permute(0, 2, 1) 
        
        # Output: (B, L, 2 * 32) = (B, L, 64)
        x, _ = self.bilstm(x)
        
        # ----------------------------------------------------
        # TDD layers require input back in the format (Batch, Channels, Length)

        # 3. Permute back for TDD: (B, L, C) -> (B, C, L)
        # x is now (B, 64, L)
        x = x.permute(0, 2, 1) 

        # --- Time Distributed Dense Layers after LSTM ---
        
        # TDD 3: (B, 64, L) -> (B, 32, L)
        x = relu(self.tdd3(x))
        
        # TDD 4: (B, 32, L) -> (B, 16, L)
        x = relu(self.tdd4(x))

        # TDD 5 (Final): (B, 16, L) -> (B, 1, L)
        # No ReLU here, as this is the final logit layer.
        x = self.tdd_final(x)
        
        # 4. Apply Sigmoid for binary classification output (0s and 1s)
        #final_output = torch.sigmoid(x)
        final_output = x
        
        return final_output