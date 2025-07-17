import torch
from torch import nn
from einops import rearrange, repeat

from modules.attention import  TemporalTransformer
# from attention import TemporalTransformer


class M3T(nn.Module):
    def __init__(self, out_dim=2, pvt_backbone=None, dim=192, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64,
                 dropout=0., emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)
        
        self.project = nn.Linear(20, 512)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.norm1 = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        b, t, _, _ = x.shape
        # print(x.shape)
        x = x.view(b, t, -1)
        # print(x.shape)
        x= self.project(x)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.dropout(x)

        # x = rearrange(x, 'b t d -> (b t) d')
        x = self.temporal_transformer(x)
        # print(x.shape)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":
    # x.shape = B, T, G, C, H, W
    x = torch.randn((1, 4, 1, 20)) # G = 1 # same granularity of wrf data, convert hourly to 15 mins
    model = M3T(out_dim=4, dim=512)
    # print(model)
    z = model(x)
    print(z)
    print(z.shape)
