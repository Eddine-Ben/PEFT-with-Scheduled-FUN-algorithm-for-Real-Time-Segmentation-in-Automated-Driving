# Suppress DeprecationWarning and UserWarning globally before any imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import json
import random
import logging
import ctypes
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
from tqdm import tqdm
import requests
import time

# 🎯 Pretty banner
BANNER = """
🎯 SegFormer++ (B0–B5) • Fisher-Guided Fine-Tuning + LoRA adapters
   • Keeps original Cityscapes resolution (no resizing) 🖼️
   • Fisher-based gradual (adapter) unfreezing 🔬🔓
   • Real-time friendly heads ⚡
"""

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from build_model_fixed import create_model
except Exception:
    create_model = None

try:
    from utils import set_random_seed as _ext_set_random_seed, create_logger as _ext_create_logger
except Exception:
    _ext_set_random_seed = None
    _ext_create_logger = None

CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES = 19

MODEL_CONFIGS = {
    "B5_HQ":  {"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,8,27,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_plusplus_hq_b5.pth"},
    "B5_FAST":{"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,8,27,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_plusplus_fast_b5.pth"},
    "B5_2X2":{"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,8,27,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_plusplus_2x2_b5.pth"},
    "B0":     {"embed_dims":[32,64,160,256],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[2,2,2,2],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_b0_cityscapes.pth"},
    "B1":     {"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[2,2,2,2],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_b1_cityscapes.pth"},
    "B2":     {"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,4,6,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_b2_cityscapes.pth"},
    "B3":     {"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,4,18,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_b3_cityscapes.pth"},
    "B4":     {"embed_dims":[64,128,320,512],"num_heads":[1,2,5,8],"mlp_ratios":[4,4,4,4],"qkv_bias":True,"depths":[3,8,27,3],"sr_ratios":[8,4,2,1],"weight_file":"weights/segformer_b4_cityscapes.pth"}
}

def create_logger(name: str, level=logging.INFO):
    if _ext_create_logger is not None:
        return _ext_create_logger(name)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def set_random_seed(seed: int = 42):
    if _ext_set_random_seed is not None:
        return _ext_set_random_seed(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def setup_cuda_optimizations():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        try:
            print(f"🚀 CUDA Device: {torch.cuda.get_device_name()} • VRAM Alloc {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        except Exception:
            print("🚀 CUDA ready")

def cleanup_memory():
    try: cv2.setNumThreads(1)
    except Exception: pass
    for _ in range(3):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc as _gc; _gc.collect()
    try: ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception: pass
    print("🧹 Memory cleanup done")

def monitor_system_resources():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated()
        total = torch.cuda.mem_get_info()[1]
        r = (used / total * 100) if total > 0 else 0
        return f"🖥️ GPU {r:.1f}% • {used/1e9:.2f}/{total/1e9:.2f} GB"
    return "🖥️ GPU N/A"

def download_file(url, save_path):
    with requests.Session() as session:
        r = session.get(url, stream=True, allow_redirects=True, timeout=60)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f, tqdm(desc=f"⬇️ {os.path.basename(save_path)}", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    pbar.update(f.write(chunk))
    print(f"✅ Downloaded: {save_path}")

# -------------------------------------------------------------------------
# LoRA for MultiheadAttention (safe for PyTorch internals)
# -------------------------------------------------------------------------
class LoRAMultiheadAttention(nn.Module):
    """
    Drop-in wrapper around nn.MultiheadAttention that injects LoRA deltas into:
      - in_proj_weight (Q/K/V) and
      - out_proj.weight
    Base weights are *frozen*; only low-rank A/B are trainable.
    """
    def __init__(self, base_mha: nn.MultiheadAttention, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base_mha, nn.MultiheadAttention)
        self.base = base_mha
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)

        # Freeze all base params
        for p in self.base.parameters():
            p.requires_grad = False

        E = self.base.embed_dim
        if self.r > 0:
            # A: (r, E)  B: (E, r)  -> B @ A  ~  (E, E)
            def _ab():
                A = nn.Parameter(torch.empty(self.r, E))
                B = nn.Parameter(torch.empty(E, self.r))
                nn.init.kaiming_uniform_(A, a=1e-2)
                nn.init.zeros_(B)
                return A, B

            self.q_A,   self.q_B   = _ab()
            self.k_A,   self.k_B   = _ab()
            self.v_A,   self.v_B   = _ab()
            self.out_A, self.out_B = _ab()
        else:
            # disabled
            self.q_A = self.q_B = self.k_A = self.k_B = self.v_A = self.v_B = self.out_A = self.out_B = None

        # copy attributes we might need
        self.batch_first = getattr(self.base, 'batch_first', False)

    def _delta(self, A: torch.Tensor, B: torch.Tensor, dtype, device):
        return (B @ A).to(dtype=dtype, device=device) if self.r > 0 else None

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):

        base = self.base
        bf = self.batch_first

        # Accept both (N,S,E) and (S,N,E) by mirroring base's batch_first
        if bf:
            if query.dim() == 3:
                query = query.transpose(0,1); key = key.transpose(0,1); value = value.transpose(0,1)

        # Choose dtype/device from base weights
        if base.in_proj_weight is not None:
            dtype  = base.in_proj_weight.dtype
            device = base.in_proj_weight.device
        else:
            dtype  = base.q_proj_weight.dtype
            device = base.q_proj_weight.device

        # Compose modified weights with LoRA deltas
        if base.in_proj_weight is not None:
            in_w = base.in_proj_weight
            out_w = base.out_proj.weight
            if self.r > 0:
                dq = self._delta(self.q_A, self.q_B, dtype, device)
                dk = self._delta(self.k_A, self.k_B, dtype, device)
                dv = self._delta(self.v_A, self.v_B, dtype, device)
                stacked = torch.cat([dq, dk, dv], dim=0)
                in_w = in_w + self.scale * stacked
                out_w = out_w + self.scale * self._delta(self.out_A, self.out_B, dtype, device)

            # Torch has slightly different signatures across versions
            try:
                attn_output, attn_weights = F.multi_head_attention_forward(
                    query, key, value,
                    embed_dim_to_check=base.embed_dim,
                    num_heads=base.num_heads,
                    in_proj_weight=in_w,
                    in_proj_bias=base.in_proj_bias,
                    bias_k=base.bias_k, bias_v=base.bias_v,
                    add_zero_attn=base.add_zero_attn,
                    dropout_p=base.dropout,
                    out_proj_weight=out_w,
                    out_proj_bias=base.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=False,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal
                )
            except TypeError:
                # Fallback for older torch without is_causal/average_attn_weights
                attn_output, attn_weights = F.multi_head_attention_forward(
                    query, key, value,
                    embed_dim_to_check=base.embed_dim,
                    num_heads=base.num_heads,
                    in_proj_weight=in_w,
                    in_proj_bias=base.in_proj_bias,
                    bias_k=base.bias_k, bias_v=base.bias_v,
                    add_zero_attn=base.add_zero_attn,
                    dropout_p=base.dropout,
                    out_proj_weight=out_w,
                    out_proj_bias=base.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=False
                )
        else:
            # Separate q/k/v projection weight path
            qp = base.q_proj_weight
            kp = base.k_proj_weight
            vp = base.v_proj_weight
            out_w = base.out_proj.weight
            if self.r > 0:
                dq = self._delta(self.q_A, self.q_B, dtype, device)
                dk = self._delta(self.k_A, self.k_B, dtype, device)
                dv = self._delta(self.v_A, self.v_B, dtype, device)
                qp = qp + self.scale * dq
                kp = kp + self.scale * dk
                vp = vp + self.scale * dv
                out_w = out_w + self.scale * self._delta(self.out_A, self.out_B, dtype, device)

            try:
                attn_output, attn_weights = F.multi_head_attention_forward(
                    query, key, value,
                    embed_dim_to_check=base.embed_dim,
                    num_heads=base.num_heads,
                    in_proj_weight=None,
                    in_proj_bias=base.in_proj_bias,
                    bias_k=base.bias_k, bias_v=base.bias_v,
                    add_zero_attn=base.add_zero_attn,
                    dropout_p=base.dropout,
                    out_proj_weight=out_w,
                    out_proj_bias=base.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=qp, k_proj_weight=kp, v_proj_weight=vp,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal
                )
            except TypeError:
                attn_output, attn_weights = F.multi_head_attention_forward(
                    query, key, value,
                    embed_dim_to_check=base.embed_dim,
                    num_heads=base.num_heads,
                    in_proj_weight=None,
                    in_proj_bias=base.in_proj_bias,
                    bias_k=base.bias_k, bias_v=base.bias_v,
                    add_zero_attn=base.add_zero_attn,
                    dropout_p=base.dropout,
                    out_proj_weight=out_w,
                    out_proj_bias=base.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=qp, k_proj_weight=kp, v_proj_weight=vp
                )

        if bf:
            attn_output = attn_output.transpose(0,1)
        return attn_output, attn_weights

def inject_lora_into_mha(root: nn.Module, rank: int = 8, alpha: int = 16):
    """Replace every nn.MultiheadAttention with LoRAMultiheadAttention."""
    wrapped = 0
    examples = []
    # Walk all named modules so we can locate parents and attributes
    for qn, m in list(root.named_modules()):
        if isinstance(m, nn.MultiheadAttention):
            # find parent and attribute name
            path = qn.split(".")
            parent = root
            for p in path[:-1]:
                parent = getattr(parent, p)
            attr = path[-1]
            new_m = LoRAMultiheadAttention(m, r=rank, alpha=alpha)
            setattr(parent, attr, new_m)
            wrapped += 1
            if len(examples) < 5:
                examples.append(qn)
    print(f"🧩 LoRA-MHA injection: wrapped {wrapped} MultiheadAttention module(s)")
    if examples:
        print("   Examples:", ", ".join(examples) + (" ..." if wrapped > 5 else ""))
    return wrapped

def count_lora_parameters(model: nn.Module):
    lora_params = 0
    total = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, LoRAMultiheadAttention):
            for p in [m.q_A, m.q_B, m.k_A, m.k_B, m.v_A, m.v_B, m.out_A, m.out_B]:
                if p is not None:
                    lora_params += p.numel()
    return lora_params, total

# -------------------------------------------------------------------------
# 🧾 Cityscapes loader (keeps original size)
# -------------------------------------------------------------------------
class CityscapesDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.img_dir = self.root / "leftImg8bit" / split
        self.mask_dir = self.root / "gtFine" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"❌ Images folder not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"❌ Masks folder not found: {self.mask_dir}")
        self.samples = []
        self.mean = torch.tensor(CITYSCAPES_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(CITYSCAPES_STD).view(3, 1, 1)
        for city in sorted(os.listdir(self.img_dir)):
            img_city_dir = self.img_dir / city
            mask_city_dir = self.mask_dir / city
            if not img_city_dir.is_dir():
                continue
            for fname in sorted(os.listdir(img_city_dir)):
                if fname.endswith("_leftImg8bit.png"):
                    img_path = img_city_dir / fname
                    mask_name = fname.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                    mask_path = mask_city_dir / mask_name
                    if mask_path.exists():
                        self.samples.append((str(img_path), str(mask_path)))
        print(f"🗂️ {split}: {len(self.samples)} samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        try:
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("image none")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError("mask none")
        except Exception:
            img = np.zeros((1024, 2048, 3), dtype=np.uint8)
            mask = np.full((1024, 2048), 255, dtype=np.uint8)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = (img - self.mean) / self.std
        mask = torch.from_numpy(mask).long()
        return img, mask

# -------------------------------------------------------------------------
# 🔩 Lightweight SegFormer head on MIT backbone (mmseg)
# -------------------------------------------------------------------------
def _local_create_model(backbone: str, out_channels: int, pretrained: bool = True) -> nn.Module:
    try:
        from mmseg.models.backbones.mit import MixVisionTransformer
    except Exception as e:
        raise ImportError("mmsegmentation required") from e
    arch = backbone.lower()
    variant = backbone.upper()
    embed_dims_map = {
        'B0':[32,64,160,256],'B1':[64,128,320,512],'B2':[64,128,320,512],'B3':[64,128,320,512],
        'B4':[64,128,320,512],'B5':[64,128,320,512],'B5_2X2':[64,128,320,512]
    }
    depths_map = {
        'b0':[2,2,2,2],'b1':[2,2,2,2],'b2':[3,4,6,3],'b3':[3,4,18,3],
        'b4':[3,8,27,3],'b5':[3,6,40,3],'b5_2x2':[3,8,27,3]
    }
    embed_dims_list = embed_dims_map.get(variant, [64,128,320,512])
    depths = depths_map.get(arch, [3,6,40,3])
    num_heads = [1,2,5,8]
    sr_ratios = [8,4,2,1]

    class Net(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            try:
                b = MixVisionTransformer(
                    embed_dims=embed_dims_list, num_heads=num_heads, mlp_ratios=[4,4,4,4],
                    qkv_bias=True, depths=depths, sr_ratios=sr_ratios,
                    drop_rate=0.0, drop_path_rate=0.1, out_indices=(0,1,2,3)
                )
            except TypeError:
                b = MixVisionTransformer(
                    embed_dims=embed_dims_list[0], num_layers=depths, num_heads=num_heads, mlp_ratio=4,
                    qkv_bias=True, sr_ratios=sr_ratios, out_indices=(0,1,2,3),
                    drop_rate=0.0, drop_path_rate=0.1
                )
            self.backbone = b
            cproj = 256
            self.proj = nn.ModuleList([nn.Conv2d(c, cproj, 1) for c in embed_dims_list])
            self.fuse = nn.Sequential(
                nn.Conv2d(cproj*4, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.cls = nn.Conv2d(256, num_classes, 1)

        def forward(self,x):
            feats = self.backbone(x)
            if isinstance(feats, dict):
                feats = [feats[k] for k in ['x1','x2','x3','x4'] if k in feats]
            h0,w0 = feats[0].shape[-2:]
            ups=[]
            for i,f in enumerate(feats):
                p = self.proj[i](f)
                if p.shape[-2:]!=(h0,w0):
                    p = F.interpolate(p,size=(h0,w0),mode='bilinear',align_corners=False)
                ups.append(p)
            y = torch.cat(ups,dim=1)
            y = self.fuse(y)
            y = F.dropout(y,0.1,training=self.training)
            y = self.cls(y)
            return y

    return Net(out_channels)

def ensure_create_model():
    if callable(create_model):
        return create_model
    def _w(backbone: str, out_channels: int, pretrained: bool = True):
        return _local_create_model(backbone=backbone, out_channels=out_channels, pretrained=pretrained)
    return _w

def _worker_init_fn(_):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

# -------------------------------------------------------------------------
# 🧠 Fisher-guided freezer/unfreezer (targets LoRA modules by default)
# -------------------------------------------------------------------------
class FisherFreezer:
    def __init__(self, model, head_keys, backbone_lr, head_lr, lora_lr=None, fisher_target='lora'):
        self.model = model
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.lora_lr = lora_lr if lora_lr is not None else backbone_lr
        self.fisher_target = fisher_target  # 'lora' or 'backbone'
        self.freezable = []
        self.frozen = []
        self.unfrozen = []
        self.head_params = []
        self.backbone_params = []
        self.fisher = {}

        # Identify head vs backbone params
        for n,p in model.named_parameters():
            if any(k in n.lower() for k in head_keys):
                self.head_params.append(p)
            else:
                self.backbone_params.append(p)

        # Freeze backbone params initially
        for p in self.backbone_params: p.requires_grad=False
        for p in self.head_params: p.requires_grad=True

        # Build freezable list (LoRA adapters grouped by module)
        for n,m in model.named_modules():
            if self.fisher_target == 'lora' and isinstance(m, LoRAMultiheadAttention):
                ps = [p for p in m.parameters(recurse=False)]  # only A/B params
                if ps:
                    self.freezable.append({'name':n,'module':m,'params':ps,'is_lora':True})
            elif self.fisher_target != 'lora':
                if any(k in n.lower() for k in ['block','attn','attention','mlp','norm','linear','layers']):
                    ps=[p for p in m.parameters(recurse=False)]
                    if ps:
                        self.freezable.append({'name':n,'module':m,'params':ps,'is_lora':False})

        self.frozen = self.freezable.copy()
        self.unfrozen = []

        print(f"🧊 Initially frozen adapter modules: {sum(1 for f in self.frozen if f.get('is_lora', False))} "
              f"• other frozen modules: {sum(1 for f in self.frozen if not f.get('is_lora', False))}")
        print(f"🔥 head trainable params: {sum(p.numel() for p in self.head_params):,}")

    def optimizer(self, weight_decay_head=1e-4, weight_decay_backbone=1e-5, weight_decay_lora=0.0):
        groups=[]
        seen=set()

        head=[p for p in self.head_params if p.requires_grad]
        if head:
            ids=[]
            for p in head:
                if id(p) not in seen:
                    ids.append(p); seen.add(id(p))
            if ids:
                groups.append({'params':ids,'lr':self.head_lr,'weight_decay':weight_decay_head,'name':'head'})

        for m in self.unfrozen:
            ps=[p for p in m['params'] if p.requires_grad]
            ids=[]
            for p in ps:
                if id(p) not in seen:
                    ids.append(p); seen.add(id(p))
            if ids:
                lr = self.lora_lr if m.get('is_lora', False) else self.backbone_lr
                wd = weight_decay_lora if m.get('is_lora', False) else weight_decay_backbone
                groups.append({'params':ids,'lr':lr,'weight_decay':wd,'name':f"unfrozen:{m['name']}"})

        if not groups:
            allp=[p for p in self.model.parameters() if p.requires_grad]
            groups=[{'params':allp,'lr':self.head_lr,'weight_decay':weight_decay_head,'name':'all'}]
        print(f"🧮 Optimizer groups: {len(groups)}")
        return optim.AdamW(groups, betas=(0.9,0.999), eps=1e-8)

    def temp_enable(self, modules):
        saved=[]
        for m in modules:
            for p in m['params']:
                saved.append((p,p.requires_grad))
                p.requires_grad=True
        return saved

    def restore(self, saved):
        for p,was in saved:
            p.requires_grad=was
        self.model.zero_grad(set_to_none=True)

    def compute_fisher(self, data_loader, device, criterion, n_samples=64, heartbeat_every=1, batch_timeout_s=None):
        if not self.frozen:
            print("✅ No frozen modules — skipping Fisher")
            return {}
        print("🔬 Computing Fisher Information …")
        saved=self.temp_enable(self.frozen)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        fisher={m['name']:0.0 for m in self.frozen}
        counts={m['name']:0 for m in self.frozen}
        seen=0
        print(f"🫀 FI heartbeat: 0/{n_samples}")
        it = iter(data_loader)
        while seen < n_samples:
            try:
                if batch_timeout_s is None:
                    imgs, masks = next(it)
                else:
                    start = time.time()
                    while True:
                        try:
                            imgs, masks = next(it); break
                        except StopIteration:
                            raise
                        except Exception:
                            time.sleep(0.01)
                        if (time.time() - start) > batch_timeout_s:
                            raise TimeoutError("FI batch fetch timeout")
                imgs=imgs.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
                logits=self.model(imgs)
                if logits.shape[-2:]!=masks.shape[-2:]:
                    logits=F.interpolate(logits,size=masks.shape[-2:],mode='bilinear',align_corners=False)
                loss=criterion(logits,masks.long())
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                for m in self.frozen:
                    s=0.0; n=0
                    for p in m['params']:
                        if p.grad is not None:
                            g=p.grad.detach()
                            s+=(g*g).sum().item(); n+=g.numel()
                    if n>0:
                        fisher[m['name']]+=s/n; counts[m['name']]+=1
                seen+=1
                if (seen % heartbeat_every) == 0:
                    print(f"🫀 FI heartbeat: {seen}/{n_samples}")
            except TimeoutError as e:
                print(f"⏳ FI loader timed out: {e} • skipping this batch"); break
            except StopIteration:
                print("ℹ️ FI loader exhausted early"); break
            except KeyboardInterrupt:
                print("🧯 FI interrupted by user"); break
            except Exception as e:
                print(f"⚠️ FI batch failed: {e} • continuing"); continue
        for k in fisher:
            fisher[k]=fisher[k]/max(1,counts[k])
        self.restore(saved)
        self.fisher=fisher
        tops = sorted(fisher.items(), key=lambda x: x[1], reverse=True)[:5]
        print("📊 Top-FI:", ", ".join([f"{n}:{s:.2e}" for n,s in tops]))
        return fisher

    def unfreeze_topk(self, k=1, threshold=0.0):
        if not self.frozen:
            print("✅ All modules already unfrozen")
            return 0
        ordered=sorted(self.fisher.items(), key=lambda x:x[1], reverse=True)
        cnt=0
        for name,score in ordered:
            if cnt>=k: break
            if score<threshold: break
            idx=None
            for i,m in enumerate(self.frozen):
                if m['name']==name: idx=i; break
            if idx is None: continue
            mod=self.frozen.pop(idx)
            for p in mod['params']: p.requires_grad=True
            self.unfrozen.append(mod)
            cnt+=1
            print(f"🔓 Unfroze: {mod['name']} (FI {score:.2e}){' [LoRA]' if mod.get('is_lora', False) else ''}")
        print(f"🧮 Frozen: {len(self.frozen)} • 🟢 Unfrozen: {len(self.unfrozen)}")
        return cnt

    def unfreeze_all(self):
        for m in self.frozen:
            for p in m['params']: p.requires_grad=True
            self.unfrozen.append(m)
        self.frozen=[]
        print("🔓 Unfroze ALL remaining target modules")

# -------------------------------------------------------------------------
# 🧪 Class names (Cityscapes)
# -------------------------------------------------------------------------
CS_CLASSES = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain','sky',
    'person','rider','car','truck','bus','train','motorcycle','bicycle'
]

def _emoji_perf(iou):
    if iou > 0.70: return "🌟 Excellent"
    if iou > 0.50: return "🟢 Good"
    if iou > 0.30: return "🟡 Fair"
    if iou > 0.10: return "🟠 Poor"
    return "🔴 Very Poor"

# -------------------------------------------------------------------------
# 🏋️ Trainer
# -------------------------------------------------------------------------
class SegFormerTrainer:
    def __init__(self, model, config, device, logger):
        self.model=model
        self.config=config
        self.device=device
        self.logger=logger
        self.best_iou=0.0
        self.current_epoch=0
        cw = torch.tensor(
            [0.8373,0.9180,0.8660,1.0345,1.0166,0.9969,0.9754,1.0489,0.8786,1.0023,0.9539,0.9843,1.1116,0.9037,1.0865,1.0955,1.0865,1.1529,1.0507],
            dtype=torch.float32, device=device
        )
        self.criterion = nn.CrossEntropyLoss(weight=cw, ignore_index=255)

        fisher_target = 'lora' if config.get('enable_lora', True) else 'backbone'
        self.freezer = FisherFreezer(
            self.model, head_keys=['classifier','cls','fuse','proj'],
            backbone_lr=config['backbone_lr'], head_lr=config['head_lr'],
            lora_lr=config.get('lora_lr', None), fisher_target=fisher_target
        )
        self.optimizer = self.freezer.optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8)
        self.early_patience = config.get('early_stopping_patience', 150)
        self.early_delta = config.get('early_stopping_min_delta', 0.01)
        self.early_restore = config.get('early_stopping_restore_best', False)
        self.patience_ctr = 0
        self.best_es = 0.0
        self.best_state = None

    def _miou_from_inter_union(self, preds, masks, num_classes=19):
        valid = masks!=255
        inters=[0]*num_classes; unions=[0]*num_classes
        for c in range(num_classes):
            p=(preds==c)&valid
            t=(masks==c)&valid
            inter=int((p&t).sum().item())
            union=int((p|t).sum().item())
            inters[c]+=inter; unions[c]+=union
        ious=[(i/u if u>0 else 0.0) for i,u in zip(inters,unions)]
        return float(np.mean(ious)), ious

    def _print_iou_report(self, epoch, inters, unions):
        class_ious = [(i/u if u>0 else 0.0) for i,u in zip(inters,unions)]
        pairs = list(zip(CS_CLASSES, class_ious, unions))
        pairs.sort(key=lambda x: x[1], reverse=True)
        present = sum(1 for _,_,u in pairs if u>0)
        poor = sum(1 for _,iou,_ in pairs if iou<=0.2)
        excel = sum(1 for _,iou,_ in pairs if iou>0.7)
        good = sum(1 for _,iou,_ in pairs if 0.5<iou<=0.7)
        print("\n📊 DETAILED IoU REPORT - Epoch", epoch)
        print("="*98)
        print(f"{'Class':<16}{'IoU':<10}{'Union Pixels':<14}{'Status':<10}{'Performance'}")
        print("-"*98)
        for cname, iou, u in pairs:
            if u>0:
                status="✅ Present"; perf=_emoji_perf(iou)
            else:
                status="❌ Absent"; perf="⚫ No Data"
            print(f"{cname:<16}{iou:<10.4f}{u:<14}{status:<10}{perf}")
        print("-"*98)
        miou_all = float(np.mean(class_ious))
        valid_ious = [iou for iou, u in zip(class_ious, unions) if u>0]
        miou_valid = float(np.mean(valid_ious)) if valid_ious else 0.0
        print(f"🎯 Standard mIoU (all 19): {miou_all:.4f}")
        print(f"✅ Valid mIoU ({present} classes): {miou_valid:.4f}")
        print(f"🌟 Excellent (>70%): {excel} • 🟢 Good (50–70%): {good} • 🔴 Poor (<20%): {poor}")
        print("="*98)
        return miou_all, class_ious, miou_valid

    def validate(self, loader):
        self.model.eval()
        loss_sum=0.0; tot_corr=0; tot_pix=0
        batch_ious=[]
        num_classes=19
        ep_inter=[0]*num_classes; ep_union=[0]*num_classes
        with torch.no_grad():
            pbar=tqdm(loader,desc="🧪 Validation", dynamic_ncols=True)
            for imgs,masks in pbar:
                imgs=imgs.to(self.device,non_blocking=True); masks=masks.to(self.device,non_blocking=True)
                logits=self.model(imgs)
                if logits.shape[-2:]!=masks.shape[-2:]:
                    logits=F.interpolate(logits,size=masks.shape[-2:],mode='bilinear',align_corners=False)
                loss=self.criterion(logits,masks.long())
                preds=torch.argmax(logits,dim=1)
                valid=masks!=255
                tot_corr+=(preds[valid]==masks[valid]).sum().item()
                tot_pix+=valid.sum().item()
                loss_sum+=loss.item()
                bi,_=self._miou_from_inter_union(preds,masks)
                batch_ious.append(bi)
                for c in range(num_classes):
                    p=(preds==c)&valid; t=(masks==c)&valid
                    ep_inter[c]+=int((p&t).sum().item())
                    ep_union[c]+=int((p|t).sum().item())
                acc=(tot_corr/tot_pix) if tot_pix>0 else 0.0
                pbar.set_postfix({'loss':f'{loss.item():.4f}','acc':f'{acc:.4f}','iou':f'{(np.mean(batch_ious) if batch_ious else 0.0):.4f}'})
        pix_acc=(tot_corr/tot_pix) if tot_pix>0 else 0.0
        avg_loss=loss_sum/max(1,len(loader))
        miou_all, class_ious, miou_valid = self._print_iou_report(self.current_epoch+1, ep_inter, ep_union)
        print(f"✅ Val • loss {avg_loss:.4f} • acc {pix_acc:.4f} • mIoU {miou_all:.4f} 💯")
        return {'val_loss':avg_loss,'pixel_accuracy':pix_acc,'mean_iou':miou_all,'class_ious':class_ious,'valid_class_miou':miou_valid,'ep_union':ep_union}

    def train_epoch(self, loader):
        self.model.train()
        loss_sum=0.0; tot_corr=0; tot_pix=0
        batch_ious=[]
        scaler = torch.amp.GradScaler('cuda') if self.config['use_amp'] else None
        accum = max(1,self.config['grad_accum_steps'])
        pbar=tqdm(loader,desc=f"🚂 Training {self.current_epoch+1}", dynamic_ncols=True)
        for i,(imgs,masks) in enumerate(pbar):
            imgs=imgs.to(self.device,non_blocking=True); masks=masks.to(self.device,non_blocking=True)
            if i%accum==0:
                self.optimizer.zero_grad(set_to_none=True)
            try:
                with (torch.amp.autocast('cuda') if self.config['use_amp'] else nullcontext()):
                    logits=self.model(imgs)
                    if logits.shape[-2:]!=masks.shape[-2:]:
                        logits=F.interpolate(logits,size=masks.shape[-2:],mode='bilinear',align_corners=False)
                    loss=self.criterion(logits,masks.long())/accum
            except RuntimeError as e:
                print(f"⚠️ Forward error: {e} • skipping batch")
                continue
            try:
                if self.config['use_amp']:
                    scaler.scale(loss).backward()
                    if (i+1)%accum==0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                        scaler.step(self.optimizer); scaler.update()
                else:
                    loss.backward()
                    if (i+1)%accum==0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                        self.optimizer.step()
            except RuntimeError as e:
                print(f"⚠️ Backward/step error: {e} • zeroing grads")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            if torch.isfinite(loss):
                loss_sum+=loss.item()*accum
            with torch.no_grad():
                preds=logits.argmax(1)
                valid=masks!=255
                tot_corr+=(preds[valid]==masks[valid]).sum().item()
                tot_pix+=valid.sum().item()
                bi,_=self._miou_from_inter_union(preds,masks)
                batch_ious.append(bi)
            pbar.set_postfix({
                'loss':f'{loss.item():.4f}',
                'acc':f'{(tot_corr/tot_pix) if tot_pix>0 else 0.0:.4f}',
                'iou':f'{(np.mean(batch_ious) if batch_ious else 0.0):.4f}',
                'lr':f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        avg_loss=loss_sum/max(1,len(loader))
        acc=(tot_corr/tot_pix) if tot_pix>0 else 0.0
        iou=float(np.mean(batch_ious)) if batch_ious else 0.0
        print(f"🧰 Train • loss {avg_loss:.4f} • acc {acc:.4f} • mIoU {iou:.4f}")
        return avg_loss,acc,iou

    def train(self, train_loader, val_loader, num_epochs):
        print("🎬 Starting training with Fisher-guided unfreezing 🔬🔓")
        best_metric=0.0
        history=[]
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        variant=self.config.get('model_variant','B5_2X2')
        logf=f"training_log_{variant}_{ts}.txt"
        with open(logf,'w') as f:
            f.write("epoch,train_loss,train_acc,train_iou,val_loss,val_acc,val_iou,lr,status\n")

        for epoch in range(num_epochs):
            self.current_epoch=epoch
            print(f"\n📈 Epoch {epoch+1}/{num_epochs} • {monitor_system_resources()}")

            if self.config['use_fisher_unfreezing']:
                warm = self.config.get('fi_warmup_batches', 0)
                if warm > 0:
                    print(f"🔥 FI warmup: reading {warm} train batches …")
                    _wseen=0
                    for _imgs,_masks in train_loader:
                        _wseen+=1
                        if _wseen>=warm: break
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                    print("🔥 FI warmup done")

                fi_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=self.config.get('fi_loader_workers', 0),
                    pin_memory=self.config.get('fi_pin_memory', False),
                    persistent_workers=False,
                    timeout=self.config.get('fi_timeout_s', 0),
                    worker_init_fn=_worker_init_fn
                )

                n_fi = (self.config.get('fisher_samples_first_epoch') if epoch == 0 else None)
                n_fi = n_fi if n_fi is not None else self.config.get('fisher_samples', 64)

                self.freezer.compute_fisher(
                    fi_loader, self.device, self.criterion,
                    n_samples=n_fi, heartbeat_every=1,
                    batch_timeout_s=self.config.get('fi_batch_fetch_timeout_s', 60)
                )
                k=self.config.get('max_modules_to_unfreeze',1)
                thr=self.config.get('fisher_threshold',0.0)
                unf=self.freezer.unfreeze_topk(k=k,threshold=thr)
                if unf>0:
                    self.optimizer=self.freezer.optimizer()
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='max', factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
                    )
            else:
                if epoch==self.config.get('freeze_backbone_epochs',10):
                    self.freezer.unfreeze_all()
                    self.optimizer=self.freezer.optimizer()
                    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='max', factor=0.9, patience=20, verbose=True, min_lr=1e-7, eps=1e-8
                    )

            tl,ta,ti=self.train_epoch(train_loader)
            vm=self.validate(val_loader)
            self.scheduler.step(vm['mean_iou'])
            lr=self.optimizer.param_groups[0]['lr']

            print("📚 LR groups:")
            for g in self.optimizer.param_groups:
                print(f"   • {g.get('name','group')} : {g['lr']:.2e}")

            with open(logf,'a') as f:
                f.write(f"{epoch+1},{tl:.6f},{ta:.6f},{ti:.6f},{vm['val_loss']:.6f},{vm['pixel_accuracy']:.6f},{vm['mean_iou']:.6f},{lr:.2e},train\n")

            if vm['mean_iou']>best_metric:
                best_metric=vm['mean_iou']; self.best_iou=best_metric
                torch.save(self.model.state_dict(), f"segformer_{variant}_finetuned_best.pth")
                print(f"💾 New best model saved! 🏆 mIoU {best_metric:.4f}")

            imp=vm['mean_iou']-self.best_es
            if imp>self.early_delta:
                self.best_es=vm['mean_iou']; self.patience_ctr=0
                if self.early_restore:
                    self.best_state={k:v.cpu().clone() for k,v in self.model.state_dict().items()}
                print(f"✨ Improved by {imp:.4f} • patience reset")
            else:
                self.patience_ctr+=1
                print(f"⏳ No improvement • patience {self.patience_ctr}/{self.early_patience}")
                if self.patience_ctr>=self.early_patience:
                    print("🛑 Early stopping")
                    if self.early_restore and self.best_state is not None:
                        self.model.load_state_dict(self.best_state)
                        torch.save(self.model.state_dict(), f"segformer_{variant}_finetuned_early_stopped.pth")
                    break

            history.append({
                'epoch':epoch+1,'train_loss':tl,'train_acc':ta,'train_iou':ti,
                'val_loss':vm['val_loss'],'val_acc':vm['pixel_accuracy'],'val_iou':vm['mean_iou'],
                'lr':lr,'frozen':len(self.freezer.frozen),'unfrozen':len(self.freezer.unfrozen)
            })

            if (epoch+1)%self.config['checkpoint_interval']==0:
                ckpt={'epoch':epoch+1,'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),'scheduler_state_dict':self.scheduler.state_dict(),'best_iou':self.best_iou,'config':self.config,'history':history,'frozen':len(self.freezer.frozen),'unfrozen':len(self.freezer.unfrozen)}
                name=f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(ckpt, name)
                print(f"🧩 Checkpoint saved: {name}")

        with open('training_history.json','w') as f:
            json.dump(history,f,indent=2)
        print(f"🏁 Training done • Best mIoU {best_metric:.4f}")
        return best_metric

# -------------------------------------------------------------------------
# 🚀 Main
# -------------------------------------------------------------------------
def main():
    print(BANNER)
    os.makedirs('weights', exist_ok=True)
    set_random_seed(42)

    # Windows: safer worker spawn
    if sys.platform.startswith("win"):
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 🔧 CONFIG
    config = {
        'model_variant': "B5_HQ",
        'dataset_root': r"C:\Users\benracho\yolov9\CDetector\SegFormer\build\Debug\SegFormerFineTune\Mask2Former\datasets\cityscapes",
        'batch_size': 1,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'num_epochs': 200,
        'checkpoint_interval': 10,
        'grad_accum_steps': 16,
        'use_amp': False,
        'warmup_epochs': 15,
        'enable_memory_cleanup': True,
        'safe_mode': True,

        # Optim LRs
        'backbone_lr': 1e-5,
        'head_lr': 3e-5,
        'lora_lr': 1e-4,                  # LoRA adapters learn a bit faster

        # Early stopping
        'early_stopping_patience': 150,
        'early_stopping_min_delta': 0.01,
        'early_stopping_restore_best': False,

        # Fisher unfreezing
        'use_fisher_unfreezing': True,
        'fisher_compute_interval': 1,
        'fisher_threshold': 0.0,
        'max_modules_to_unfreeze': 1,
        'unfreeze_interval': 1,
        'fisher_samples': 64,
        'freeze_all_initially': True,

        # DataLoader knobs
        'num_workers_train': 2,
        'num_workers_val': 2,
        'persistent_workers': False,
        'prefetch_factor': 2,

        # FI robustness knobs (Windows-safe)
        'fi_use_dedicated_loader': True,
        'fi_loader_workers': 0,
        'fi_pin_memory': False,
        'fi_timeout_s': 0,
        'fi_batch_fetch_timeout_s': 60,
        'fi_warmup_batches': 2,
        'fisher_samples_first_epoch': 8,

        # 🔧 LoRA settings
        'enable_lora': True,
        'lora_rank': 8,
        'lora_alpha': 16,
    }

    root = Path(config['dataset_root'])
    expected_imgs = root / "leftImg8bit" / "train"
    expected_lbls = root / "gtFine" / "train"
    if not expected_imgs.exists() or not expected_lbls.exists():
        raise FileNotFoundError(f"❌ Cityscapes path looks wrong:\n{root}\nExpected: {expected_imgs}\n          {expected_lbls}")

    logger = create_logger('segformer_fisher_training')
    setup_cuda_optimizations()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧭 Device: {device}")

    print("🧹 Pre-flight cleanup …")
    cleanup_memory()

    variant = config['model_variant']
    model_cfg = MODEL_CONFIGS.get(variant if variant in MODEL_CONFIGS else variant.upper(), MODEL_CONFIGS['B5_2X2'])
    weight_path = model_cfg.get('weight_file', 'weights/segformer.pth')

    # Backbone name mapping
    if variant.upper().startswith('B5'):
        backbone_name = 'b5_2x2' if variant.upper()=='B5_2X2' else 'b5'
    elif variant.upper() in ['B0','B1','B2','B3','B4']:
        backbone_name = variant.lower()
    else:
        backbone_name='b5'

    _create = ensure_create_model()
    model = _create(backbone=backbone_name, out_channels=NUM_CLASSES, pretrained=not os.path.exists(weight_path))

    # Conservative init
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if getattr(m,'weight',None) is not None: nn.init.constant_(m.weight,1)
            if getattr(m,'bias',None) is not None: nn.init.constant_(m.bias,0)

    model.apply(init_weights)
    print("🪄 Applied conservative weight init")

    # Optional partial weight load (Cityscapes MIT weights)
    if not os.path.exists(weight_path):
        urls={
            "B5_HQ":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
            "B5_FAST":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
            "B5_2X2":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
            "B0":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
            "B1":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth",
            "B2":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth",
            "B3":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8aba8.pth",
            "B4":"https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth"
        }
        key = variant if variant in urls else "B5_2X2"
        try:
            print(f"🌐 Pretrained weights missing — downloading for {variant} …")
            download_file(urls[key], weight_path)
        except Exception as e:
            print(f"⚠️ Download failed: {e} • continuing with random/ImageNet init")

    if os.path.exists(weight_path):
        try:
            try:
                ckpt=torch.load(weight_path,map_location='cpu',weights_only=True)
            except Exception:
                ckpt=torch.load(weight_path,map_location='cpu')
            sd=ckpt['state_dict'] if isinstance(ckpt,dict) and 'state_dict' in ckpt else ckpt
            msd=model.state_dict()
            filt={}
            for k,v in sd.items():
                mk=k
                if mk not in msd and not mk.startswith('backbone.'):
                    mk=f'backbone.{mk}'
                if mk in msd and msd[mk].shape==v.shape:
                    filt[mk]=v
            model.load_state_dict(filt,strict=False)
            print(f"📦 Loaded {len(filt)} compatible params from checkpoint")
        except Exception as e:
            print(f"⚠️ Weight load failed: {e} • using fresh init")

    # 🔌 Inject LoRA into MultiheadAttention ONLY (safe)
    if config.get('enable_lora', True):
        wrapped = inject_lora_into_mha(model, rank=config.get('lora_rank', 8), alpha=config.get('lora_alpha', 16))
        lora_p, total_p = count_lora_parameters(model)
        print(f"🧮 Params: total {total_p/1e6:.2f}M • LoRA {lora_p/1e6:.3f}M ({100.0*lora_p/max(1,total_p):.2f}%)")

    model=model.to(device)

    # Datasets / Loaders (no resizing anywhere)
    train_dataset=CityscapesDataset(root=config['dataset_root'],split='train')
    val_dataset  =CityscapesDataset(root=config['dataset_root'],split='val')

    train_loader=DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers_train'],
        pin_memory=True,
        persistent_workers=config['persistent_workers'],
        drop_last=True,
        prefetch_factor=(config['prefetch_factor'] if config['num_workers_train']>0 else None),
        worker_init_fn=_worker_init_fn
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers_val'],
        pin_memory=True,
        persistent_workers=config['persistent_workers'],
        drop_last=False,
        prefetch_factor=(config['prefetch_factor'] if config['num_workers_val']>0 else None),
        worker_init_fn=_worker_init_fn
    )

    # Sanity check
    print("🔎 Sanity check forward …")
    model.eval()
    with torch.no_grad():
        sample_img, _ = train_dataset[0]
        h,w = sample_img.shape[-2], sample_img.shape[-1]
        dummy=torch.randn(1,3,h,w,device=device)
        out=model(dummy)
        if torch.isnan(out).any():
            print("🚨 NaNs detected in output! Re-init…")
            model.apply(lambda m:None)
        else:
            print(f"✅ Model OK • output shape: {tuple(out.shape)}")
    model.train()

    trainer=SegFormerTrainer(model,config,device,logger)
    print("🏋️ Starting training …")
    best=trainer.train(train_loader,val_loader,config['num_epochs'])
    print(f"🏆 Best mIoU: {best:.4f}")

if __name__ == "__main__":
    main()
