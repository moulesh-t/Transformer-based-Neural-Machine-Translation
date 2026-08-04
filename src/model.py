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
        
    def forward(self, x: torch.Tensor) -> Tensor:
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
    
    
class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism.
    """

    def __init__(self, d_model :int, num_heads :int, dropout :float = 0.1) -> None:
        """
        Initializes the Multi-Head Attention class for computing attention scores.
        
        Args:
            d_model (int): The dimension of input embeddings.
            num_heads (int): The number of heads used for attention.
            dropout (float): The dropout rate.
            
        Returns:
            None
        """
        super().__init__()
        assert d_model % num_heads == 0, "Embedding Dimension must be divisble by number of heads"
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask = None) -> Tensor:
        """
        Implements the scaled dot product attention mechanism.
        Args:
            Q (tensor): Query vector of shape (batch_size, seq_len, num_heads, n_k)
            K (tensor): Query vector of shape (batch_size, seq_len, num_heads, n_k)
            v (tensor): Query vector of shape (batch_size, seq_len, num_heads, n_k)
            mask (tensor or None): Masking Tensor (Default is None)
        
        Returns:
            attn_scores (tensor): The attention scores of shape (batch_size, seq_len, num_heads, n_k)
        """
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return self.dropout(attn_weights) @ V
    
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask = None) -> Tensor:
        """
        Implements the complete forward pass of Attention Mechanism
        
        Args:
            Q (tensor): Query vector of shape (batch_size, max_len, d_model)
            K (tensor): Query vector of shape (batch_size, max_len, d_model)
            V (tensor): Query vector of shape (batch_size, max_len, d_model)
            mask (tensor or None): Masking Tensor (Default is None)
        
        Returns:
            output (tensor): the output tensor of shape (batch_size, max_len, d_model)
        """
        Q = self.W_q(Q).reshape(Q.shape[0], Q.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, max_len, n_k)
        K = self.W_k(K).reshape(K.shape[0], K.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, max_len, n_k)
        V = self.W_v(V).reshape(V.shape[0], V.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, max_len, n_k)
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().reshape(output.shape[0], output.shape[1], -1)
        output = self.W_o(output) # (batch_size, max_len, d_model)
        return output


class FeedForward(nn.Module):
    """
    Implements the feed forward neural network of AI Model
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Initializes the feed forward network.

        Args:
            d_model: Embedding Dimension.
            d_ff: Expansion factor for the hidden layer.
            dropout: The dropout rate.
            
        Returns:
            None
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the feed-forward network to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            output (tensor): the output tensor of shape (batch_size, max_len, d_model)
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x
    
    
class EncoderLayer(nn.Module):
    """
    Implements the Encoder layer of the Transformer
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1) -> None:
        """
        Initializes the Transformer encoder layer.

        Args:
            d_model: Dimension of the input and output embeddings.
            num_heads: Number of attention heads.
            d_ff: Expansion factor for the feed-forward network.
            dropout: Dropout probability applied after each sublayer.
            
        Returns: 
            None
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask = None) -> Tensor:
        """
        Applies forward pass of Encoder Layer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.
            
        Returns:
            output tensor of shape (batch_size, seq_len, d_model).
        """
        attn_output = self.attention(x, x, x, mask)
        output = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(output)
        output = self.ln2(output + self.dropout(ffn_output))
        return output


class DecoderLayer(nn.Module):
    """
    Implements the Decoder layer of the Transformer
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1) -> None:
        """
        Initializes the Transformer's Decoder layer.
        
        Args:
            d_model (int): Dimension of the input and output embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Expansion factor for the feed-forward network.
            dropout (float): Dropout probability applied after each sublayer.
                    
        Returns: 
            None
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, encoder_output: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
        """
        Implements the forward pass in the Decoder layer
        
        Args:
            x (tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            encoder_output (tensor): Output of the enocder layer of shape (batch_size, src_seq_len, d_model).
            src_mask: Optional attention mask for source sentences.
            tgt_mask: Optional attention mask for target sentences.
                    
        Returns:
            output tensor of shape (batch_size, seq_len, d_model).
        """
        self_attn_scores = self.self_attention(x, x, x, tgt_mask)
        x = self.ln1(x + self.dropout(self_attn_scores))
        # Encoder - Decoder Cross Attention
        cross_attn_scores = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.ln2(x + self.dropout(cross_attn_scores))
        ffn_outputs = self.ffn(x)
        output = self.ln3(x + self.dropout(ffn_outputs))
        return output
            