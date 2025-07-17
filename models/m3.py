"""
Fixed and runnable Vision Transformer code
"""
from typing import List, Tuple, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
import torch.nn.functional as F

# Fixed attention modules
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MultiModalPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, ctx, ts):
        return self.fn(self.norm_src(ctx), self.norm_tgt(ts))

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """
        Args:
            x: Sequence of shape [B, N, D]
        """
        q = self.to_q(x)
        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn

class MultiModalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, src, tgt):
        q = self.to_q(tgt)
        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn

# Transformer classes
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C]
        """
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(self, src: torch.Tensor):
        """
        Args:
            src: Source sequence of shape [B, N, D]
        """
        attention_scores = {}
        for i in range(len(self.blocks)):
            sattn, sff = self.blocks[i]

            out, sattn_scores = sattn(src)
            attention_scores[f"self_attention_layer_{i}"] = sattn_scores  # Fixed: unique keys
            src = out + src
            src = sff(src) + src

        return src, attention_scores

class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.multimodal_layers = nn.ModuleList([])

        for _ in range(depth):
            self.multimodal_layers.append(
                nn.ModuleList(
                    [
                        MultiModalPreNorm(
                            dim,
                            MultiModalAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            src: Source sequence of shape [B, N, D] (image patches)
            tgt: Target sequence of shape [B, M, D] (time series)
        """
        attention_scores = {}
        for i in range(len(self.multimodal_layers)):
            cattn, cff = self.multimodal_layers[i]
            out, cattn_scores = cattn(src, tgt)
            attention_scores[f"multimodal_attention_layer_{i}"] = cattn_scores  # Fixed: unique keys
            tgt = out + tgt
            tgt = cff(tgt) + tgt

        return tgt, attention_scores

class M3(nn.Module):
    def __init__(
        self,
        image_size: Union[List[int], Tuple[int]] = [128, 128],
        patch_size: Union[List[int], Tuple[int]] = [16, 16],
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 4,
        ctx_channels: int = 1,  # Changed to match your input
        ts_channels: int = 20,  # Changed to match your input
        ts_length: int = 4,     # Changed to match your input
        out_dim: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_mlp_heads: int = 1,
        use_glu: bool = True,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_dim_head: int = 128,
    ):
        super().__init__()
        self.ctx_channels = ctx_channels
        self.ts_channels = ts_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_mlp_heads = num_mlp_heads

        # Patch Embedding Setup
        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        patch_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]
        num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            ),
            nn.Linear(patch_dim, dim),
        )

        # Positional Encoding Setup
        self.ts_embedding = nn.Linear(self.ts_channels, dim)

        # Vision transformer for processing context
        self.ctx_encoder = VisionTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            use_glu,
        )

        # Set up position encoding
        self.pe_ctx = nn.Parameter(torch.randn(1, num_patches, dim))
        # print(f"Positional encoding shape: {self.pe_ctx.shape}")
        self.pe_ts = nn.Parameter(torch.randn(1, 1, dim))

        # MultiModal-Model Components   
        self.mixer = MultiModalTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            use_glu,
        )

        # Time series encoder and decoder
        self.ts_encoder = Transformer(
            dim,
            ts_length,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_enctodec = nn.Linear(dim, decoder_dim)
        self.temporal_transformer = Transformer(
            decoder_dim,
            ts_length,
            decoder_depth,
            decoder_heads,
            decoder_dim_head,
            decoder_dim * mlp_ratio,
            dropout=dropout,
        )

        # Multiple MLP heads for different outputs
        self.mlp_heads = nn.ModuleList([])
        for i in range(num_mlp_heads):
            self.mlp_heads.append(
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, out_dim, bias=True),
                    nn.ReLU(),
                )
            )

    def forward(self, ctx: torch.Tensor, ts: torch.Tensor):
        """
        Args:
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
        """
        B, T, _, H, W = ctx.shape

        # Process context frames
        ctx = rearrange(ctx, "b t c h w -> (b t) c h w")
        # print(f"ctx shape before patch embedding: {ctx.shape}")
        ctx = self.to_patch_embedding(ctx)  # [BT, N, D]
        # print(f"ctx shape after patch embedding: {ctx.shape}")
        ctx = ctx + self.pe_ctx  # Fixed: Add positional embedding
        
        latent_ctx, self_attention_scores = self.ctx_encoder(ctx)
        
        # print(f"latent_ctx shape after encoder: {latent_ctx.shape}")

        # Process time series
        # print(f"ts shape before embedding: {ts.shape}")
        ts = self.ts_embedding(ts)
        # print(f"ts shape after embedding: {ts.shape}")
        
        # print(f"ts shape before encoder: {ts.shape}")
        latent_ts = self.ts_encoder(ts)
        # print(f"latent_ts shape after encoder: {latent_ts.shape}")
        
        latent_ts = rearrange(latent_ts, "b t c -> (b t) c").unsqueeze(1)
        latent_ts = latent_ts + self.pe_ts  # Fixed: Add positional embedding
        
        # print(f"latent_ts shape after positional encoding: {latent_ts.shape}")

        # MultiModal attention
        latent_ts, multimodal_attention_scores = self.mixer(latent_ctx, latent_ts)
        latent_ts = latent_ts.squeeze(1)
        latent_ts = self.ts_enctodec(rearrange(latent_ts, "(b t) c -> b t c", b=B))

        # Final temporal processing
        y = self.temporal_transformer(latent_ts)

        # Multiple MLP heads
        outputs = []
        for i in range(self.num_mlp_heads):
            mlp = self.mlp_heads[i]
            output = mlp(y)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=2)

        return (outputs, self_attention_scores, multimodal_attention_scores)

# Test function for your specific shapes
def test_model_with_your_shapes():
    """
    Test the model with your exact input shapes:
    - ctx: [1, 4, 1, 128, 128] (1 batch, 4 frames, 1 channel, 128x128)
    - ts: [1, 4, 20] (1 batch, 4 timesteps, 20 features)
    """
    print("Creating model for your shapes...")
    
    model = M3(
        image_size=[128, 128],    # 128x128 images
        patch_size=[16, 16],      # 16x16 patches -> 64 patches per image
        ctx_channels=1,           # Grayscale images
        ts_channels=20,           # 20 time series features
        ts_length=4,              # 4 timesteps
        dim=128,                  # Model dimension
        depth=2,                  # 2 layers (reduced for testing)
        heads=4,                  # 4 attention heads
        mlp_ratio=4,              # MLP expansion ratio
        dropout=0.1,              # Dropout
        num_mlp_heads=1,          # Single output head
        decoder_dim=128,          # Decoder dimension
        decoder_depth=2,          # Decoder depth
    )
    
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create your exact input tensors
    ctx = torch.randn(4, 4, 1, 128, 128)  # Your context shape
    ts = torch.randn(4, 4, 20)            # Your time series shape
    
    print("\nInput shapes:")
    print(f"  ctx: {list(ctx.shape)} (batch, time, channels, height, width)")
    print(f"  ts: {list(ts.shape)} (batch, time, features)")
    
    print("\nTracing through model...")
    

    outputs, self_attn, multimodal_attn = model(ctx, ts)
    print(f"Output shape: {list(outputs.squeeze().shape)} (batch, time, heads, features)")
            



def run_complete_test():
    """Run the complete test with analysis"""
    print("🚀 RUNNING COMPLETE TEST")
    print("="*60)
    
    # Test with your shapes
    test_model_with_your_shapes()


# Run the complete test
if __name__ == "__main__":
    results = run_complete_test()