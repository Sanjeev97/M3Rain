import torch
from torch import nn

from models.iTransformer import Model as iTransformer
# from iTransformer import Model as iTransformer

class iTnM3(nn.Module):
    def __init__(self, out_dim=2, dim=192, batch_size=64, pool='cls'):
        super().__init__()

        self.batch_size = batch_size
        self.iTransformer = iTransformer(seq_len=out_dim, pred_len=out_dim, d_model_tn=dim, embed='timeF', freq='t', dropout_tn=0.1,
                                         factor_tn=1, e_layers_tn=3, d_ff_tn=512, n_heads_pt=8,
                                         activation_pt='gelu')

    def forward(self, x):
        b, t, _, _ = x.shape
        
        x = x.view(b, t, -1)
        x = self.iTransformer(x)
        x = x[:, :, -1]
        return x


if __name__ == "__main__":

    ys = torch.randn((32, 4, 1, 20)) 
    model = iTnM3(out_dim=4, dim=512)
    z = model(x=ys)
    print(z.shape)
