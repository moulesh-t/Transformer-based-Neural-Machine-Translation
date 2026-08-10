import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Transformer
from optimizer import AdamW
from scheduler import WarmupCosineScheduler
import logging
import wandb

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
    scaler = torch.amp.GradScaler(enabled=device.type=='cuda')
    model.to(device)
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    wandb.init(project='AI_Translator', config={
        'epochs': epochs,
        'batch_size': dataloader.batch_size,
    })
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                logits = model(src, tgt_input, src_mask, tgt_mask).transpose(1,2)
                loss = loss_function(logits, tgt[:, 1:])
            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if step % 100 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.6f}"
                wandb.log({"step_loss": loss.item(), "lr": scheduler.get_lr()})
                print(msg)
                logging.info(msg)
        if (epoch+1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch+1, total_loss/len(dataloader), f'./models/{epoch+1}.pt')
        msg = f"Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(dataloader):.4f}"
        wandb.log({"epoch_loss": total_loss/len(dataloader), "epoch": epoch+1})
        print(msg)
        logging.info(msg)
        
    torch.save(model.state_dict(), './models/final_model.pt')
    