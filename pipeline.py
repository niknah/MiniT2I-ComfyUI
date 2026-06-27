import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x):
    x1, x2 = x.reshape(*x.shape[:-1], 2, -1).unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_channels=3, pca_channels=128, hidden_size=1248):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        hidden_dim = (hidden_features + 7) // 8 * 8
        self.w1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w3 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TextRotaryEmbedding1D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x):
        b, length, h, d = x.shape
        inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        pos = torch.arange(length, device=x.device, dtype=torch.float32)
        angles = torch.einsum("l,f->lf", pos, inv)
        angles = torch.cat([angles, angles], dim=-1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        return x * cos[None, :, None, :] + rotate_half(x) * sin[None, :, None, :]


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = head_dim // 2
        self.theta = theta

    def forward(self, x):
        length = x.shape[1]
        side = int(math.sqrt(length))
        if side * side != length:
            raise ValueError(f"image token length must be square, got {length}")
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)[: self.dim // 2] / self.dim)
        )
        t = torch.arange(side, device=x.device, dtype=torch.float32)
        base = torch.einsum("l,f->lf", t, freqs)
        f_h, f_w = torch.broadcast_tensors(base[:, None, :], base[None, :, :])
        angles = torch.cat([f_h, f_w], dim=-1)
        angles = torch.cat([angles, angles], dim=-1).reshape(length, -1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        return x * cos[None, :, None, :] + rotate_half(x) * sin[None, :, None, :]


class MultiModalRotaryEmbeddingFast(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.text_rope = TextRotaryEmbedding1D(head_dim)
        self.vision_rope = VisionRotaryEmbeddingFast(head_dim)

    def forward(self, x, txt_len: int):
        txt = self.text_rope(x[:, :txt_len])
        img = self.vision_rope(x[:, txt_len:])
        return torch.cat([txt, img], dim=1)


class PlainTextTransformerBlock(nn.Module):
    def __init__(self, hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.attn_proj = nn.Linear(inner_dim, hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = TextRotaryEmbedding1D(head_dim)

    def forward(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * (self.head_dim ** -0.5)
        out = torch.einsum("bhqk,bkhd->bqhd", attn.softmax(dim=-1), v).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class DoubleStreamDiTBlock(nn.Module):
    def __init__(self, hidden_size=1248, txt_hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.hidden_size = hidden_size
        self.txt_hidden_size = txt_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(txt_hidden_size)
        self.txt_norm2 = RMSNorm(txt_hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.txt_qkv = nn.Linear(txt_hidden_size, inner_dim * 3)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = MultiModalRotaryEmbeddingFast(head_dim)
        self.img_attn_proj = nn.Linear(inner_dim, hidden_size)
        self.txt_attn_proj = nn.Linear(inner_dim, txt_hidden_size)
        self.img_mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLUMlp(txt_hidden_size, int(txt_hidden_size * mlp_ratio))

    def forward(self, x, txt, vec):
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = self.img_norm1(x)
        txt_norm = self.txt_norm1(txt)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q = self.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt)
        k = self.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt)
        v = torch.cat([v_t, v_i], dim=1)
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * (self.head_dim ** -0.5)
        out = torch.einsum("bhqk,bkhd->bqhd", attn.softmax(dim=-1), v)
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, -1))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class FinalLayer(nn.Module):
    def __init__(self, hidden_size=1248, patch_size=16, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, vec=None):
        return self.linear(self.norm_final(x))


def get_2d_sincos_pos_embed(embed_dim, grid_size, device, dtype):
    grid_h = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


@dataclass
class MMJiTConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3
    txt_input_size: int = 1024
    hidden_size: int = 768
    txt_hidden_size: int = 768
    cond_vec_size: int = 768
    depth_double: int = 17
    txt_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    prompt_length: int = 256
    n_T: int = 100
    prediction: str = "x"
    sampler: str = "euler"
    cfg_channels: int = 3
    cfg_interval: tuple = (0.0, 1.0)
    llm: str = "google/flan-t5-large"


class MMJiT(nn.Module):
    def __init__(self, cfg: MMJiTConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_img_size = cfg.image_size // cfg.patch_size
        self.img_embedder = BottleneckPatchEmbed(
            cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.pca_channels, cfg.hidden_size
        )
        self.txt_embedder = nn.Linear(cfg.txt_input_size, cfg.txt_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.txt_input_size))
        self.t_embedder = TimestepEmbedder(cfg.cond_vec_size)
        self.pooled_embedder = nn.Linear(cfg.txt_input_size, cfg.cond_vec_size, bias=False)
        self.txt_preamble_blocks = nn.ModuleList(
            [PlainTextTransformerBlock(cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio) for _ in range(cfg.txt_preamble_depth)]
        )
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamDiTBlock(
                    cfg.hidden_size, cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio
                )
                for _ in range(cfg.depth_double)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)

    def unpatchify(self, x):
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(b, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(b, c, h * p, w * p)

    def forward(self, img, t, context, attn_mask):
        if img.ndim == 4 and img.shape[1] != self.cfg.in_channels:
            img = img.permute(0, 3, 1, 2)
        attn_mask = attn_mask.to(device=context.device)
        context = torch.where(attn_mask[:, :, None] > 0.5, context, self.mask_token.to(dtype=context.dtype))
        x = self.img_embedder(img)
        pos = get_2d_sincos_pos_embed(self.cfg.hidden_size, self.latent_img_size, x.device, x.dtype)
        x = x + pos[None]
        t_vec = self.t_embedder(t)
        txt = self.txt_embedder(context.to(dtype=self.txt_embedder.weight.dtype))
        pooled_text = context.mean(dim=1)
        vec = t_vec + self.pooled_embedder(pooled_text.to(dtype=self.pooled_embedder.weight.dtype))
        for block in self.txt_preamble_blocks:
            txt = block(txt)
        for block in self.double_blocks:
            x, txt = block(x, txt, vec)
        combined = torch.cat([txt, x], dim=1)
        out = self.final_layer(combined, vec)
        img_out = out[:, txt.shape[1] :, :]
        return self.unpatchify(img_out)


class DiffusionModel(nn.Module):
    def __init__(self, cfg: Optional[MMJiTConfig] = None):
        super().__init__()
        self.cfg = cfg or MMJiTConfig()
        self.net = MMJiT(self.cfg)

    def real_t_to_embed_t(self, t):
        return t

    def pred_velocity(self, x, t, text, mask):
        x0 = self.net(x, self.real_t_to_embed_t(t), text, mask)
        return (x0 - x) / torch.clamp(1 - t[:, None, None, None], min=0.001)

    def cfg_velocity(self, x, t, text, mask, cfg_scale: float):
        b = x.shape[0]
        xx = torch.cat([x, x], dim=0)
        tt = torch.cat([t, t], dim=0)
        yy = torch.cat([text, text], dim=0)
        mm = torch.cat([mask, torch.zeros_like(mask)], dim=0)
        out = self.pred_velocity(xx, tt, yy, mm)
        cond, uncond = out[:b], out[b:]
        use_cfg = ((t >= self.cfg.cfg_interval[0]) & (t <= self.cfg.cfg_interval[1])).to(out.dtype)
        scale = torch.where(use_cfg[:, None, None, None] > 0, torch.tensor(cfg_scale, device=x.device, dtype=out.dtype), torch.tensor(1.0, device=x.device, dtype=out.dtype))
        return uncond + (cond - uncond) * scale

    @torch.no_grad()
    def sample(self, text, mask, cfg_scale=6.0, generator=None, progress=False):
        b = text.shape[0]
        device = text.device
        dtype = next(self.parameters()).dtype
        x = torch.randn(
            b, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size,
            generator=generator, device=device, dtype=dtype,
        ) * 2
        timesteps = torch.linspace(0.0, 1.0, self.cfg.n_T + 1, device=device, dtype=dtype)
        iterator = range(self.cfg.n_T)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator)
        for i in iterator:
            t_cur = timesteps[i].expand(b)
            t_next = timesteps[i + 1].expand(b)
            v = self.cfg_velocity(x, t_cur, text.to(dtype), mask.to(dtype), cfg_scale)
            x = x + (t_next - t_cur)[:, None, None, None] * v
        return x


import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from diffusers import DiffusionPipeline, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


transformers_logging.set_verbosity_error()


class MiniT2IFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        if train_t_schedule not in {"uniform", "lognorm"}:
            raise ValueError(f"Unsupported train_t_schedule: {train_t_schedule}")

    def sample_train_timesteps(self, batch_size, device, dtype=torch.float32, generator=None):
        if self.config.train_t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        normal = normal * self.config.t_lognorm_sigma + self.config.t_lognorm_mu
        return torch.sigmoid(normal).to(dtype=dtype)

    def get_inference_timesteps(self, num_inference_steps=None, device=None, dtype=torch.float32):
        steps = int(num_inference_steps or self.config.num_inference_steps)
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)


class MiniT2IMMJiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        txt_input_size: int = 1024,
        hidden_size: int = 768,
        txt_hidden_size: int = 768,
        cond_vec_size: int = 768,
        depth_double: int = 17,
        txt_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6666666666666665,
        pca_channels: int = 128,
        prompt_length: int = 256,
        n_T: int = 100,
        prediction: str = "x",
        sampler: str = "euler",
        cfg_channels: int = 3,
        cfg_interval: tuple = (0.0, 1.0),
        llm: str = "google/flan-t5-large",
    ):
        super().__init__()
        cfg = MMJiTConfig(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            txt_input_size=txt_input_size,
            hidden_size=hidden_size,
            txt_hidden_size=txt_hidden_size,
            cond_vec_size=cond_vec_size,
            depth_double=depth_double,
            txt_preamble_depth=txt_preamble_depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            pca_channels=pca_channels,
            prompt_length=prompt_length,
            n_T=n_T,
            prediction=prediction,
            sampler=sampler,
            cfg_channels=cfg_channels,
            cfg_interval=tuple(cfg_interval),
            llm=llm,
        )
        self.model = DiffusionModel(cfg)

    @property
    def mmjit_config(self) -> MMJiTConfig:
        return self.model.cfg

    def forward(self, img, t, context, attn_mask):
        return self.model.net(img, t, context, attn_mask)

    def pred_velocity(self, x, t, text, mask):
        return self.model.pred_velocity(x, t, text, mask)

    def sample(self, text, mask, cfg_scale=6.0, generator=None, progress=False):
        return self.model.sample(text, mask, cfg_scale=cfg_scale, generator=generator, progress=progress)


class MiniT2ITextToImagePipeline(nn.Module):
    def __init__(
        self,
        transformer: MiniT2IMMJiTModel,
        scheduler: Optional[MiniT2IFlowMatchScheduler] = None,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        if not isinstance(scheduler, MiniT2IFlowMatchScheduler):
            scheduler = MiniT2IFlowMatchScheduler(
                train_t_schedule=train_t_schedule,
                t_lognorm_mu=t_lognorm_mu,
                t_lognorm_sigma=t_lognorm_sigma,
                num_inference_steps=num_inference_steps,
            )
        self.transformer = transformer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.config = SimpleNamespace(
            text_encoder_name=text_encoder_name,
            train_t_schedule=scheduler.config.train_t_schedule,
            t_lognorm_mu=scheduler.config.t_lognorm_mu,
            t_lognorm_sigma=scheduler.config.t_lognorm_sigma,
            num_inference_steps=scheduler.config.num_inference_steps,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        torch_dtype: Optional[torch.dtype] = None,
        text_encoder_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        root = Path(pretrained_model_name_or_path)
        if not root.exists():
            root = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        transformer = MiniT2IMMJiTModel.from_pretrained(root / "transformer", torch_dtype=torch_dtype, **kwargs)
        scheduler_dir = root / "scheduler"
        if scheduler_dir.exists():
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()
        text_encoder_name = transformer.mmjit_config.llm
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype=text_encoder_dtype,
            local_files_only=local_files_only,
        )
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=text_encoder_name,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(save_directory / "transformer", **kwargs)
        self.scheduler.save_pretrained(save_directory / "scheduler")

    def _encode_prompt(self, prompt: Union[str, List[str]], device):
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        cfg = self.transformer.mmjit_config
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.prompt_length,
        )
        input_ids = tokens.input_ids.to(device)
        attn = tokens.attention_mask.to(device)
        text = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return text, attn

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
    ):
        device = next(self.transformer.parameters()).device
        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for p in prompt:
                prompt_batch.extend([p] * num_images_per_prompt)

        old_steps = self.transformer.mmjit_config.n_T
        self.transformer.model.cfg.n_T = int(num_inference_steps or self.scheduler.config.num_inference_steps)
        try:
            text, attn = self._encode_prompt(prompt_batch, device)
            model_dtype = next(self.transformer.parameters()).dtype
            images = self.transformer.sample(
                text.to(dtype=model_dtype),
                attn.to(dtype=model_dtype),
                cfg_scale=guidance_scale,
                generator=generator,
                progress=progress,
            )
        finally:
            self.transformer.model.cfg.n_T = old_steps


        if output_type == "pt":
            images = (images.clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).to(torch.float16).cpu()
            pass
        else:
            images = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            if output_type == "pil":
                images = [Image.fromarray(image) for image in images]
                
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


class MiniT2IPipeline(DiffusionPipeline):
    MODEL_ALIASES = {
        "b": "minit2i-b-16",
        "b16": "minit2i-b-16",
        "b-16": "minit2i-b-16",
        "base": "minit2i-b-16",
        "minit2i-b16": "minit2i-b-16",
        "minit2i-b-16": "minit2i-b-16",
        "minit2i-b/16": "minit2i-b-16",
        "l": "minit2i-l-16",
        "l16": "minit2i-l-16",
        "l-16": "minit2i-l-16",
        "large": "minit2i-l-16",
        "minit2i-l16": "minit2i-l-16",
        "minit2i-l-16": "minit2i-l-16",
        "minit2i-l/16": "minit2i-l-16",
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def _resolve_model_type(cls, model_type: str) -> str:
        key = model_type.lower().replace("_", "-")
        if key not in cls.MODEL_ALIASES:
            choices = ", ".join(sorted(set(cls.MODEL_ALIASES)))
            raise ValueError(f"Unknown model_type={model_type!r}. Expected one of: {choices}")
        return cls.MODEL_ALIASES[key]

    @staticmethod
    def _resolve_root(
        repo_id_or_path: Union[str, os.PathLike],
        model_dir: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, os.PathLike]],
        local_files_only: bool,
    ) -> Path:
        root = Path(repo_id_or_path)
        if root.exists():
            return root
        return Path(
            snapshot_download(
                repo_id=str(repo_id_or_path),
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=[
                    f"{model_dir}/transformer/*",
                    f"{model_dir}/scheduler/*",
                ],
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        model_type: str = "b16",
        repo_id_or_path: Union[str, os.PathLike] = "MiniT2I/MiniT2I",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        text_encoder_dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        model_dir = self._resolve_model_type(model_type)
        root = self._resolve_root(repo_id_or_path, model_dir, revision, cache_dir, local_files_only)
        model_root = root / model_dir
        transformer = MiniT2IMMJiTModel.from_pretrained(model_root / "transformer", torch_dtype=torch_dtype)
        scheduler = MiniT2IFlowMatchScheduler.from_pretrained(model_root / "scheduler")
        text_encoder_name = transformer.mmjit_config.llm
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype=text_encoder_dtype,
            local_files_only=local_files_only,
        )
        pipe = MiniT2ITextToImagePipeline(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=text_encoder_name,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        return pipe(prompt=prompt, **kwargs)


def build_transformer_from_checkpoint(ckpt_path: Union[str, os.PathLike]) -> MiniT2IMMJiTModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = MMJiTConfig(**payload["config"])
    transformer = MiniT2IMMJiTModel(**asdict(cfg))
    prefixed = payload["state_dict"]
    state_dict = {}
    for key, value in prefixed.items():
        if key.startswith("net."):
            state_dict[f"model.{key}"] = value
        else:
            state_dict[f"model.{key}"] = value
    transformer.load_state_dict(state_dict, strict=True)
    return transformer
