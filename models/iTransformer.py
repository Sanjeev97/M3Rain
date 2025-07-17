import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, seq_len, pred_len, d_model_tn, embed, freq, dropout_tn,
                 factor_tn, e_layers_tn, d_ff_tn, n_heads_pt, activation_pt):
        
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = False
        # self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model_tn, embed, freq,
                                                    dropout_tn)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor_tn, attention_dropout=dropout_tn,
                                      output_attention=self.output_attention), d_model_tn, n_heads_pt),
                    d_model_tn,
                    d_ff_tn,
                    dropout=dropout_tn,
                    activation=activation_pt
                ) for l in range(e_layers_tn)
            ],
            norm_layer=torch.nn.LayerNorm(d_model_tn)
        )
        self.projector = nn.Linear(d_model_tn, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        self.use_norm = True  # Assuming normalization is always used in this model
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
if __name__ == "__main__":


    # Assuming you have the Model class imported
    # from your_model_file import Model

    # Test configuration
    batch_size = 4
    seq_len = 96      # Input sequence length
    pred_len = 24     # Prediction length  
    num_variates = 7  # Number of variables/features

    # Model parameters
    d_model_tn = 512
    embed = 'timeF'
    freq = 'h'
    dropout_tn = 0.1
    factor_tn = 1
    e_layers_tn = 2
    d_ff_tn = 2048
    n_heads_pt = 8
    activation_pt = 'gelu'

    print("Creating model...")
    model = Model(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model_tn=d_model_tn,
        embed=embed,
        freq=freq,
        dropout_tn=dropout_tn,
        factor_tn=factor_tn,
        e_layers_tn=e_layers_tn,
        d_ff_tn=d_ff_tn,
        n_heads_pt=n_heads_pt,
        activation_pt=activation_pt
    )

    # Create ONLY the main time series data
    x_enc = torch.randn(batch_size, seq_len, num_variates)  # [B, L, N]
    x_mark_enc = None  # No time features

    print(f"Input shape: {x_enc.shape}")
    print(f"Time features: {x_mark_enc}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc)

    print(f"Output shape: {output.shape}")
    print("Success! Model works without time features.")    