import torch
import torch.nn as nn
from model import Transformer
from data import Tokenizer


def translate(model: Transformer, sentence: str, tokenizer: Tokenizer, device: torch.device, max_len: int= 50) -> str:
    """
    Translate function for inference.
    """
    model.eval()
    src = torch.LongTensor([tokenizer.encode(sentence)]).to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        encoder_output = model.encoder(model.pos_embed(model.src_embed(src)), src_mask)
    tgt = [tokenizer.sos_id]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tokens = torch.LongTensor([tgt]).to(device)
            tgt_len = tgt_tokens.size(1)
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool().to(device)
            decoder_output = model.decoder(
                model.pos_embed(model.tgt_embed(tgt_tokens)),
                encoder_output,
                src_mask,
                causal_mask
            )
            logits = model.proj(decoder_output).to(device)
            next_token = logits[0, -1, :].argmax().item()
            if next_token == tokenizer.eos_id:
                break
            tgt.append(next_token)
    return tokenizer.decode(tgt[1:])