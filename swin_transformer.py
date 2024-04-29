import torch
from torch import nn, einsum
import numpy as np

from einops import rearrange, repeat

import torch.nn.functional as f


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        # check dims value
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x), **kwargs)   # post norm for version 2


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement *
             window_size] = float('-inf')  # down left section
        mask[:-displacement * window_size, -displacement *
             window_size:] = float('-inf')  # up right section

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2',
                         h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)]))

    distances = indices[None, :, :] - indices[:, None, :]

    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1, self.heads, 1, 1, 1))), requires_grad=True) # 1 h 1 1 1

        self.cpb_mlp = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(2, 64, bias=True),
                                nn.ReLU(),
                                nn.Linear(64, heads, bias=True))
        
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False) # (w_h w_w) (w_h w_w)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False) # (w_h w_w) (w_h w_w)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        relative_coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w] ,indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0) # 1 2*w_h-1 2*w_w-1 2
        
        if self.window_size != 1:
            relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        self.relative_coords_table = nn.Parameter(torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8), requires_grad=False) # 1 2*w_h-1 2*w_w-1 2

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w] ,indexing='ij'))  # 2 w_h w_w
        coords_flatten = torch.flatten(coords, 1)  # 2 (w_h w_w)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2 (w_h w_w) (w_h w_w)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (w_h w_w) (w_h w_w) 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords.sum(-1) # (w_h w_w) (w_h w_w)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        _, n_h, n_w, _, h = *x.shape, self.heads # B, H, W, C, heads

        qkv = self.to_qkv(x).chunk(3, dim=-1) # B, H, W, C_in
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d', # b h (nw_h nw_w) (w_h w_w) d
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = (f.normalize(q, dim=-1) @ f.normalize(k, dim=-1).transpose(-2, -1)) # b h (nw_h nw_w) (w_h w_w) (w_h w_w)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01,device=x.device))).exp()
        dots = dots * logit_scale # b h (nw_h nw_w) (w_h w_w) (w_h w_w)

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, h) # (2*w_h-1 2*w_w-1) h
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size * self.window_size, self.window_size * self.window_size, -1).unsqueeze(0).unsqueeze(0) # 1 1 (w_h w_w) (w_h w_w) h
        relative_position_bias = relative_position_bias.permute(0, 4, 1, 2, 3).contiguous() # 1 h 1 (w_h w_w) (w_h w_w)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias) # 1 h 1 (w_h w_w) (w_h w_w)
        

        dots = dots + relative_position_bias # b h (nw_h nw_w) (w_h w_w) (w_h w_w)

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1) # b h (nw_h nw_w) (w_h w_w) (w_h w_w)

        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v) # b h (nw_h nw_w) (w_h w_w) d
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)', # B, H, W, C_in
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out) # B, H, W, C

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out
    
class WindowAttention_old(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5  # scale = √head_dim
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # Version 2
        self.tau = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)

            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(
                window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(
                2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        # Dot product similarity (Version 1)
        # dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        # Version 2
        # Normalize q and k w.r.t. each row
        q = f.normalize(q, p=2, dim=-1)
        k = f.normalize(k, p=2, dim=-1)

        # Cosine similarity (Version 2)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) / self.tau

        if self.relative_pos_embedding:
            tmp1 = self.relative_indices[:, :, 0]
            dots += self.pos_embedding[self.relative_indices[:,
                                                             :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size)))
        self.mlp_block = Residual(
            PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.patch_merge = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=downscaling_factor,
                                     stride=downscaling_factor,
                                     padding=0)

    def forward(self, x):
        x = self.patch_merge(x).permute(0, 2, 3, 1)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging_Conv(in_channels=in_channels, out_channels=hidden_dimension,
                                                 downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, head_dim=32, window_size=7,   # channels = 1 for panchromatic images and we got to change the
                 downscaling_factors=(4, 2, 2, 2)):                                                                        # window_size
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size)

    def forward(self, img):
        x = self.stage1(img)
        P3 = self.stage2(x)
        P4 = self.stage3(P3)
        P5 = self.stage4(P4)
        return P3, P4, P5


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=3, window_size=7, **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, channels=channels, window_size=window_size, **kwargs)


# net = swin_t(channels=1, window_size=8)
# dummy_x = torch.randn(2, 1, 512, 512)
# logits = net(dummy_x)
# for i in logits:
#     print(i.shape)