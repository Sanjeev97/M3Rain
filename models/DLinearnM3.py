import torch
from torch import nn

from models.DLinear import Model as DLinear
# from DLinear import Model as DLinear

class DLinearnM3(nn.Module):
    def __init__(self, out_dim=2, dim=192, batch_size=64):
        super().__init__()

        self.batch_size = batch_size
        self.dlinear = DLinear(seq_len=out_dim, pred_len=out_dim, dec_in=dim, individual=False)

    def forward(self, x, ys=None, yl=None):
        b, t, _, _ = x.shape
        x = x.view(b, t, -1)
        x = self.dlinear(x)
        x = x[:, :, -1]
        return x


if __name__ == "__main__":
    ys = torch.randn((32, 4, 1, 20))
    model = DLinearnM3(out_dim=4, dim=512)
    # print(model)
    z = model(x=ys)
    # print(z)
    print(z.shape)
