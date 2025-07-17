import torch
from torch import nn

from models.PatchTST import Model as PatchTST
# from PatchTST import Model as PatchTST

class PatchTSTnM3(nn.Module):
    def __init__(self, out_dim=2, dim=192, batch_size=64, pool='cls'):
        super().__init__()

        self.batch_size = batch_size
        self.patchTST = PatchTST(seq_len=out_dim, pred_len=out_dim, dec_in=20, d_model_tn=dim, dropout_tn=0.1,
                                         factor_tn=3, e_layers_tn=2, n_heads_pt=16,
                                         activation_pt='gelu')

    def forward(self, x):
        b, t, _, _ = x.shape
        
        x = x.view(b, t, -1)
        x = self.patchTST(x)
        x = x[:, :, -1]
        return x


if __name__ == "__main__":

    ys = torch.randn((32, 4, 1, 20)) # G = 1 # same granularity of wrf data, convert hourly to 15 mins

    model = PatchTSTnM3(out_dim=4, dim=512)
    z = model(x=ys)
    print(z.shape)
