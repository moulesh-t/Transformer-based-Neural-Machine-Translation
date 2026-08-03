import torch
from torch import Tensor
import torch.nn as nn

class PositionalEncoding(torch.nn.Module):
    """
    A torch.nn.Module class for Positional Encodings
    """
    
    def __init__(self, d_model: int, max_len: int=5000, dropout: float=0.1) -> None:
        """
        Initializes the Positional Encoding class from torch.nn.Module class.
            
        Args:
            d_model (int): The dimension of input embeddings.
            max_len (int): The maximum length of input sequence.
            dropout (float): The dropout rate.
            
        Returns:
            None
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)) # (d_model / 2, )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32) # (max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the positional encodings on input data tensor
        
        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, max_len, d_model)
            
        Returns: 
            tensor of shape (batch_size, max_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :] # type: ignore
        x = self.dropout(x)
        return x
