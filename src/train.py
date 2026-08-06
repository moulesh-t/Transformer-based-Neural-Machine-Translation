import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Transformer
from optimizer import AdamW
from scheduler import WarmupCosineScheduler
import logging

logging.basicConfig(
    filename='training.log',
    level = logging.INFO,
    format= '%(asctime)s - %(message)s'
)


def save_checkpoint(model: Transformer, optimizer: AdamW, epoch: int, loss: float, path: str) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def train(model: Transformer, dataloader: DataLoader, optimizer: AdamW, scheduler:  WarmupCosineScheduler, device: torch.device, epochs: int) -> None:
    """
    Implement Training of Transformer Model.
    """
    model.to(device)
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(epochs):
        total_loss = 0
        for step, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_input = tgt[:, :-1]
            tgt_len = tgt_input.size(1)
            tgt_pad_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
            tgt_mask = (tgt_pad_mask & causal_mask).to(device)
            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask).transpose(1,2)
            loss = loss_function(logits, tgt[:, 1:])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.6f}"
                print(msg)
                logging.info(msg)
        if (epoch+1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch+1, total_loss, f'./models/{epoch+1}.pt')
        msg = f"Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(dataloader):.4f}"
        print(msg)
        logging.info(msg)
    torch.save(model.state_dict(), './models/final_model.pt')
    