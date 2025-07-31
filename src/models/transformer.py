import copy

import torch.nn as nn

from .attention import MultiHeadAttention
from .embeddings import Embeddings, PositionalEncoding
from .encoder_decoder import Encoder, Decoder, EncoderLayer, DecoderLayer, Generator, PositionwiseFeedForward


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device='cpu'):  # Added device arg
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout).to(device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
    position_encoding = PositionalEncoding(d_model, dropout).to(device)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(device), N).to(device),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(device), N).to(device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(device), c(position_encoding)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(device), c(position_encoding)),
        Generator(d_model, tgt_vocab).to(device)
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)
