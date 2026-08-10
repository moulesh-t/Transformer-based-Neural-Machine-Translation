import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
from data import Tokenizer, TranslationDataset
from model import Transformer
from optimizer import AdamW
from scheduler import WarmupCosineScheduler
from train import train
from evaluate import evaluate
from inference import translate


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
)

df = pd.read_csv('./data/dataset_clean.tsv', sep='\t')
df = df[['english', 'german']].dropna()
pairs = list(zip(df['english'], df['german']))
pairs = pairs[:10000]
tokenizer = Tokenizer('./data/tokenizer.model')
vocab_size = tokenizer.vocab_size()
dataset = TranslationDataset(pairs, tokenizer, 50)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, 32, True, num_workers=0)
val_loader = DataLoader(val_dataset, 32, num_workers=0)
model = Transformer(
                vocab_size,
                vocab_size,
                256,
                8,
                3,
                512,
                50,
                0.15
                )
epochs = 10
optimizer = AdamW(model.parameters(), 3e-4, weight_decay=0.01) #type:ignore
total_steps = epochs * len(train_loader)
warmup_steps = total_steps // 10
lr_scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
train(model, train_loader, optimizer, lr_scheduler, device, epochs)
val_loss = evaluate(model, val_loader, device)
print(f"Val Loss: {val_loss:.4f}")

s1 = 'Hi, how are you!'
s2 = "It's so good to meet you."
s3 = "I've been waiting for you."

print(translate(model, s1, tokenizer, device, 50))
print(translate(model, s2, tokenizer, device, 50))
print(translate(model, s3, tokenizer, device, 50))
