import math
import torch
import torch.nn as nn
from inspect import isfunction
from einops import rearrange
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import requests


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        fnx = self.fn(x, *args, **kwargs)
        if isinstance(fnx, tuple):
            return fnx[0] + x, *fnx[1:]
        elif isinstance(fnx, torch.Tensor):
            return fnx + x
        else:
            raise TypeError


class DiffusionEmbedding(nn.Module):
    def __init__(self, pos_emb_scale, dim):
        super().__init__()
        self.scale = pos_emb_scale  # 50000
        # self.dim = dim
        half_dim = dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / float(half_dim)
        exponents = 1e-4 ** exponents
        self.register_buffer('exponents', exponents)

    # noise_level: [B]
    def forward(self, noise_level):
        # print(noise_level.shape, self.exponents.shape, self.exponents.unsqueeze(0).shape)
        exponents = self.exponents.repeat((noise_level.shape[0], 1))
        # print(noise_level.unsqueeze(-1).shape, exponents.shape)
        # print(exponents)
        x = noise_level.unsqueeze(-1) * exponents * self.scale
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # [B, self.dim]
        # print(x.shape)
        return x


class SinusoidalPosEmb(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv1d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules
class ConvNextBlock1D(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True,
                 ft_container=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)

        layer_norm = LayerNorm(dim) if norm else nn.Identity()
        if norm and ft_container is not None:
            ft_container.append(layer_norm)

        self.net = nn.Sequential(
            layer_norm,
            nn.Conv1d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim_out * mult, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1')  # modified

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, length = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c (x)', h=self.heads), qkv)
        q = q * self.scale

        kb = k
        k = k.softmax(dim=-1)
        # print("k shape", k.shape)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        out = rearrange(out, 'b h c (x) -> b (h c) x', h=self.heads, x=length)
        return self.to_out(out), (q, kb)


class LinearAttention2(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, length = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c (x)', h=self.heads), qkv)
        q = q * self.scale

        attn_scores = torch.einsum('b h l d, b h n d -> b h l n', q, k)
        weights = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', weights, v)
        out = rearrange(out, 'b h c (x) -> b (h c) x', h=self.heads, x=length)
        return self.to_out(out), weights


def visualize_attention_weights(qk, title="Attention weights"):
    q, k = qk
    attn_scores = torch.einsum('b h l d, b h n d -> b h l n', q, k)
    # weights = attn_scores
    weights = F.softmax(attn_scores, dim=-1)

    num_heads = weights.shape[1]

    attn_weights = weights.detach().cpu().numpy()  # Convert to numpy for visualization

    # Assuming we visualize the first head and first example
    fig, axs = plt.subplots(1, num_heads, figsize=(20, 5))
    for i in range(num_heads):
        sns.heatmap(attn_weights[0, i], ax=axs[i], cmap='viridis')
        axs[i].set_title(f'Head {i}')
        axs[i].set_xlabel('Key')
        axs[i].set_ylabel('Query')
    plt.tight_layout()
    plt.show()


# Main Model

class UNet1D(nn.Module):
    def __init__(
            self,
            dim,
            sequence_length=480,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            output_mean_scale=False,
            residual=False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)
        self.output_mean_scale = output_mean_scale
        self.sequence_length = sequence_length
        self.draw_attn = False

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print("Unet dims:", dims, in_out)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                DiffusionEmbedding(50000, dim),
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim),
                nn.SiLU()
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.fine_tunes = []

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            res = Residual(PreNorm(dim_out, LinearAttention(dim_out)))
            ds = Downsample(dim_out) if not is_last else nn.Identity()
            self.fine_tunes.append(res)
            self.fine_tunes.append(ds)

            self.downs.append(nn.ModuleList([
                ConvNextBlock1D(dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0,
                                ft_container=self.fine_tunes),
                ConvNextBlock1D(dim_out, dim_out, time_emb_dim=time_dim,
                                ft_container=self.fine_tunes),
                res,
                ds
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim,
                                          ft_container=self.fine_tunes)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.fine_tunes.append(self.mid_attn)
        # self.mid_attn = nn.Identity()
        self.mid_block2 = ConvNextBlock1D(mid_dim, mid_dim, time_emb_dim=time_dim,
                                          ft_container=self.fine_tunes)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            res = Residual(PreNorm(dim_in, LinearAttention(dim_in)))
            us = Upsample(dim_in) if not is_last else nn.Identity()
            self.fine_tunes.append(res)
            self.fine_tunes.append(us)

            self.ups.append(nn.ModuleList([
                ConvNextBlock1D(dim_out * 2, dim_in, time_emb_dim=time_dim,
                                ft_container=self.fine_tunes),
                ConvNextBlock1D(dim_in, dim_in, time_emb_dim=time_dim,
                                ft_container=self.fine_tunes),
                res,
                us
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock1D(dim, dim),
            nn.Conv1d(dim, out_dim, 1),
            # nn.Tanh() # ADDED
        )
        self.fine_tunes.append(self.final_conv)

        # self.final_fc = nn.Linear(sequence_length, sequence_length)

    def forward(self, output_noisy, condition=None, t=None, noise_level=None, time=None):
        # print(x.shape, time.shape if time is not None else "")
        if condition is not None:
            x = torch.cat([output_noisy, condition], 1)
        else:
            x = output_noisy

        vis = None
        if self.draw_attn:
            vis = Visualizer()
            vis.draw_power(output_noisy, condition)

        orig_x = x
        t = None
        if noise_level is not None and exists(self.time_mlp):
            noise_level = noise_level.squeeze(1)
            t = self.time_mlp(noise_level)

        original_mean = torch.mean(x, [1, 2], keepdim=True)
        h = []

        fmaps = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            if vis is not None:
                fmaps.append(x)
            x, _ = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if vis is not None:
            fmaps.append(x)

        x, qk = self.mid_attn(x)

        # visualize_attention_weights(x, "conved")
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            # skip connection is h
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            if vis is not None:
                fmaps.append(x)
            x, _ = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        if vis is not None:
            vis.visualize(fmaps)
            visualize_attention_weights(qk, "weights")
            self.draw_attn = False

        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1, 2], keepdim=True)
            out = out - original_mean + out_mean

        # print(out.shape)
        # out = self.final_fc(out)

        return out

    def freeze(self, freeze=True):
        req_grad = not freeze

        for param in self.parameters():
            param.requires_grad = req_grad

        for layer in self.fine_tunes:
            print(type(layer))
            for param in layer.parameters():
                param.requires_grad = True


# Process the input image
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(requests.get(image_path, stream=True).raw)
    image = transform(image).unsqueeze(0)
    return image


# Visualize the feature maps
def visualize_feature_maps(model, image):
    feature_maps = model(image)
    for layer_num, fmap in enumerate(feature_maps):
        plt.figure(figsize=(15, 15))
        layer_fmaps = fmap.squeeze(0)
        for i in range(min(layer_fmaps.size(0), 16)):  # Displaying first 16 feature maps
            plt.subplot(4, 4, i + 1)  # 4x4 grid for the feature maps
            plt.imshow(layer_fmaps[i].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
        plt.show()


class Visualizer:
    def __init__(self):
        plt.figure(figsize=(15, 15))
        self.plt_idx = 1

    def draw_power(self, app, mains):
        print(app.shape, mains.shape)

        plt.subplot(4, 4, self.plt_idx)
        plt.plot(mains[0][0].detach().cpu().numpy(), linewidth=1, color="dimgray")
        plt.plot(app[0][0].detach().cpu().numpy(), linewidth=1)
        plt.axis('off')

        self.plt_idx += 1

    def visualize(self, feature_maps):
        # for i in [0, 2, 4, 6]:
        for i in range(len(feature_maps)):
            layer_fmaps = feature_maps[i]
            print(layer_fmaps.shape)
            plt.subplot(4, 4, self.plt_idx)
            plt.imshow(layer_fmaps[0].detach().cpu().numpy())
            # plt.title(f'{}')
            plt.axis('off')
            self.plt_idx += 1

    def show(self):
        plt.show()
