import torch
import torch.nn as nn
from model import Transformer
from torch.utils.data import DataLoader

def evaluate(model: Transformer, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the performance of the model.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_input = tgt[:, :-1]
            tgt_len = tgt_input.size(1)
            tgt_pad_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
            tgt_mask = (tgt_pad_mask & causal_mask).to(device)
            logits = model(src, tgt_input, src_mask, tgt_mask).transpose(1,2)
            loss = loss_fn(logits, tgt[:, 1:])
            total_loss += loss.item()
    total_loss = total_loss / len(dataloader)
    return total_loss