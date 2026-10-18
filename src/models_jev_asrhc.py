"""
models_jev_asrhc.py
===================
Novel Architecture: Anatomical Strip-Routed ConvNeXt with Source-Prior Harmonic
Calibration (ASR-HC) and Joint Execution Vectorization (JEV) Shared-Stem Engine.

UPAR Challenge (WACV 2027) - Track 1: Pedestrian Attribute Recognition.

Architecture Components:
------------------------
1. ConvNeXt-Base Shared Backbone:
   - Stem and Stages 0..2 are strictly frozen.
   - Stage 4 (the 4th stage, 1024 channels) is trainable and spatially sliced into
     3 horizontal anatomical strips:
       * Strip 0 (Top):    Head / Shoulder region
       * Strip 1 (Middle): Torso / Upper-Body region
       * Strip 2 (Bottom): Lower-Body / Footwear region
2. AnatomicalStripHead (ASR-HC Head):
   - Anatomical Strip Slicing with spatial pooling.
   - Cross-Strip Attention Token Refinement Layer:
     Multi-head self-attention across the 3 anatomical strip tokens with learnable
     anatomical positional embeddings, LayerNorm, and FFN.
   - Anatomically Routed Attribute Classifier:
     Routes refined strip tokens to corresponding attribute subsets matching
     the official 40 UPAR Track 1 attributes (10 Head, 15 Torso, 15 Lower-Body).
   - Global contextual fusion pathway to harmonize cross-strip dependencies.
3. BaselineGAPHead:
   - Standard Global Average Pooling baseline head with LayerNorm, Dropout,
     and linear projection for benchmark comparison.
4. JEVMultiSplitEngine:
   - Joint Execution Vectorization inference engine holding the shared stem/backbone.
   - Vectorized evaluation of multiple heads ("asr", "baseline", "both") in a single
     forward pass without redundant backbone computation.
   - Vectorized multi-split/domain routing (Market1501, PA-100K, PETA) for high-throughput
     inference and unified representation extraction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    timm = None
    _HAS_TIMM = False

# ==============================================================================
# CANONICAL UPAR TRACK 1 ATTRIBUTE SPECIFICATION (40 Attributes)
# ==============================================================================

UPAR_40_ATTRIBUTES: Tuple[str, ...] = (
    "Age-Young",                        # 0
    "Age-Adult",                        # 1
    "Age-Old",                          # 2
    "Gender-Female",                    # 3
    "Hair-Length-Short",                # 4
    "Hair-Length-Long",                 # 5
    "Hair-Length-Bald",                 # 6
    "UpperBody-Length-Short",           # 7
    "UpperBody-Color-Black",            # 8
    "UpperBody-Color-Blue",             # 9
    "UpperBody-Color-Brown",            # 10
    "UpperBody-Color-Green",            # 11
    "UpperBody-Color-Grey",             # 12
    "UpperBody-Color-Orange",           # 13
    "UpperBody-Color-Pink",             # 14
    "UpperBody-Color-Purple",           # 15
    "UpperBody-Color-Red",              # 16
    "UpperBody-Color-White",            # 17
    "UpperBody-Color-Yellow",           # 18
    "UpperBody-Color-Other",            # 19
    "LowerBody-Length-Short",           # 20
    "LowerBody-Color-Black",            # 21
    "LowerBody-Color-Blue",             # 22
    "LowerBody-Color-Brown",            # 23
    "LowerBody-Color-Green",            # 24
    "LowerBody-Color-Grey",             # 25
    "LowerBody-Color-Orange",           # 26
    "LowerBody-Color-Pink",             # 27
    "LowerBody-Color-Purple",           # 28
    "LowerBody-Color-Red",              # 29
    "LowerBody-Color-White",            # 30
    "LowerBody-Color-Yellow",           # 31
    "LowerBody-Color-Other",            # 32
    "LowerBody-Type-Trousers&Shorts",   # 33
    "LowerBody-Type-Skirt&Dress",       # 34
    "Accessory-Backpack",               # 35
    "Accessory-Bag",                    # 36
    "Accessory-Glasses-Normal",         # 37
    "Accessory-Glasses-Sun",            # 38
    "Accessory-Hat",                    # 39
)

# Anatomical partition indices matching pedestrian morphology
STRIP_HEAD_INDICES: List[int] = [0, 1, 2, 3, 4, 5, 6, 37, 38, 39]  # 10 attributes
STRIP_TORSO_INDICES: List[int] = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 35, 36]  # 15 attributes
STRIP_LOWER_INDICES: List[int] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # 15 attributes

# Group definitions for competitive evaluation and analysis
AGE_GROUP_INDICES: List[int] = [0, 1, 2]
HAIR_GROUP_INDICES: List[int] = [4, 5, 6]
UPPER_COLOR_INDICES: List[int] = list(range(8, 20))  # 12 colors
LOWER_COLOR_INDICES: List[int] = list(range(21, 33))  # 12 colors
LOWER_TYPE_INDICES: List[int] = [33, 34]


# ==============================================================================
# CROSS-STRIP ATTENTION REFINEMENT LAYER
# ==============================================================================

class CrossStripAttentionRefiner(nn.Module):
    """
    Cross-strip attention token refinement layer.
    
    Transforms the 3 localized anatomical tokens (Head, Torso, Lower-Body) into
    contextually refined representations via multi-head self-attention with
    residual connections, anatomical positional embeddings, and feed-forward MLP.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_strips: int = 3,
    ) -> None:
        super().__init__()
        self.num_strips = num_strips
        self.embed_dim = embed_dim
        
        # Learnable anatomical positional embeddings for Head, Torso, Lower-Body
        self.strip_pos_embed = nn.Parameter(torch.zeros(1, num_strips, embed_dim))
        nn.init.trunc_normal_(self.strip_pos_embed, std=0.02)
        
        # Pre-LayerNorm multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = nn.Dropout(dropout)
        
        # Pre-LayerNorm Feed-Forward Network
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = nn.Dropout(dropout)

    def forward(
        self,
        strip_tokens: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            strip_tokens: Tensor of shape (B, num_strips, embed_dim)
            return_attn_weights: whether to return attention map
            
        Returns:
            refined_tokens: Tensor of shape (B, num_strips, embed_dim)
            attn_weights: Optional Tensor of shape (B, num_strips, num_strips)
        """
        # Inject anatomical identity embeddings
        x = strip_tokens + self.strip_pos_embed
        
        # Self-attention block across strips
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.cross_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            need_weights=return_attn_weights,
            average_attn_weights=True,
        )
        x = x + self.drop_path1(attn_out)
        
        # FFN block
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        
        return x, attn_weights


# ==============================================================================
# ANATOMICAL STRIP HEAD (ASR-HC)
# ==============================================================================

class AnatomicalStripHead(nn.Module):
    """
    Anatomical Strip-Routed Head with Cross-Strip Attention Refinement.
    
    Processing Workflow:
    1. Slices the Stage 4 spatial feature map F in R^(B x C x H x W) horizontally
       into 3 anatomical strips:
         - Strip 0: Head / Shoulder (top ~1/3 of H)
         - Strip 1: Torso / Upper-Body (middle ~1/3 of H)
         - Strip 2: Lower-Body (bottom ~1/3 of H)
    2. Spatially pools each strip to form anatomical tokens in R^(B x C).
    3. Refines the tokens via CrossStripAttentionRefiner (cross-strip context exchange).
    4. Routes refined tokens to specialized sub-classifiers:
         - Head classifier: 10 logits (Age, Gender, Hair, Glasses, Hat)
         - Torso classifier: 15 logits (UpperBody length & 12 colors, Backpack, Bag)
         - Lower classifier: 15 logits (LowerBody length & 12 colors, Trousers, Skirt)
    5. Combines routed predictions with holistic fused representation into
       canonical 40-attribute logits in R^(B x 40).
    """
    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 40,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
        
        # Slicing indices
        self.head_indices = STRIP_HEAD_INDICES
        self.torso_indices = STRIP_TORSO_INDICES
        self.lower_indices = STRIP_LOWER_INDICES
        
        num_head_attrs = len(self.head_indices)    # 10
        num_torso_attrs = len(self.torso_indices)  # 15
        num_lower_attrs = len(self.lower_indices)  # 15
        assert num_head_attrs + num_torso_attrs + num_lower_attrs == num_classes, (
            f"Anatomical partitioning count ({num_head_attrs + num_torso_attrs + num_lower_attrs}) "
            f"must equal total classes ({num_classes})"
        )
        
        # Spatial pooling per strip
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Cross-strip attention token refiner
        self.token_refiner = CrossStripAttentionRefiner(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            num_strips=3,
        )
        
        # Specialized anatomical sub-classifiers
        self.head_norm = nn.LayerNorm(in_channels)
        self.head_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_head_attrs),
        )
        
        self.torso_norm = nn.LayerNorm(in_channels)
        self.torso_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_torso_attrs),
        )
        
        self.lower_norm = nn.LayerNorm(in_channels)
        self.lower_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_lower_attrs),
        )
        
        # Holistic cross-strip fusion classifier
        self.fusion_norm = nn.LayerNorm(in_channels * 3)
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels * 3, num_classes),
        )

    def slice_anatomical_strips(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Slices the Stage 4 spatial feature map horizontally into 3 anatomical regions:
        Head/Shoulder (top), Torso (middle), and Lower-Body (bottom).
        Handles any spatial height H >= 3 with balanced boundary partitioning.
        """
        B, C, H, W = feature_map.shape
        if H < 3:
            # Fallback for degenerate tiny spatial resolution
            t = self.pool(feature_map).flatten(1)
            return t, t, t
        
        h1 = int(round(H / 3.0))
        h2 = int(round(2.0 * H / 3.0))
        
        # Ensure non-empty slices
        h1 = max(1, min(h1, H - 2))
        h2 = max(h1 + 1, min(h2, H - 1))
        
        strip_head = feature_map[:, :, 0:h1, :]
        strip_torso = feature_map[:, :, h1:h2, :]
        strip_lower = feature_map[:, :, h2:, :]
        
        token_head = self.pool(strip_head).flatten(1)   # (B, C)
        token_torso = self.pool(strip_torso).flatten(1) # (B, C)
        token_lower = self.pool(strip_lower).flatten(1) # (B, C)
        
        return token_head, token_torso, token_lower

    def forward(
        self,
        feature_map: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Args:
            feature_map: Tensor in R^(B x 1024 x H x W) from Stage 4
            return_details: whether to return intermediate strip tokens and attention
            
        Returns:
            logits: Tensor in R^(B x 40)
            details (optional): Dict of intermediate representations
        """
        B, C, H, W = feature_map.shape
        
        # 1. Slicing and initial pooling
        t_head, t_torso, t_lower = self.slice_anatomical_strips(feature_map)
        raw_tokens = torch.stack([t_head, t_torso, t_lower], dim=1)  # (B, 3, C)
        
        # 2. Cross-strip attention token refinement
        refined_tokens, attn_weights = self.token_refiner(
            raw_tokens,
            return_attn_weights=return_details,
        )  # (B, 3, C)
        
        r_head = refined_tokens[:, 0, :]   # (B, C)
        r_torso = refined_tokens[:, 1, :]  # (B, C)
        r_lower = refined_tokens[:, 2, :]  # (B, C)
        
        # 3. Anatomically routed specialized classification
        logits_head = self.head_classifier(self.head_norm(r_head))    # (B, 10)
        logits_torso = self.torso_classifier(self.torso_norm(r_torso))# (B, 15)
        logits_lower = self.lower_classifier(self.lower_norm(r_lower))# (B, 15)
        
        # 4. Assemble routed logits in canonical 40-attribute order
        out_dtype = logits_head.dtype
        routed_logits = torch.zeros(B, self.num_classes, device=feature_map.device, dtype=out_dtype)
        routed_logits[:, self.head_indices] = logits_head
        routed_logits[:, self.torso_indices] = logits_torso.to(out_dtype)
        routed_logits[:, self.lower_indices] = logits_lower.to(out_dtype)
        
        # 5. Cross-strip holistic fused pathway
        fused_tokens = refined_tokens.reshape(B, 3 * C)  # (B, 3C)
        fused_logits = self.fusion_classifier(self.fusion_norm(fused_tokens)).to(out_dtype)  # (B, 40)
        
        # 6. Harmonized total logits
        fusion_w = self.fusion_weight.to(out_dtype)
        total_logits = routed_logits + fusion_w * fused_logits
        
        if return_details:
            details = {
                "raw_tokens": raw_tokens,
                "refined_tokens": refined_tokens,
                "attn_weights": attn_weights,
                "routed_logits": routed_logits,
                "fused_logits": fused_logits,
                "token_head": r_head,
                "token_torso": r_torso,
                "token_lower": r_lower,
            }
            return total_logits, details
        
        return total_logits


# ==============================================================================
# BASELINE GLOBAL AVERAGE POOLING HEAD
# ==============================================================================

class BaselineGAPHead(nn.Module):
    """
    Standard Global Average Pooling (GAP) Baseline Classification Head.
    
    Serves as the empirical baseline to isolate and validate the performance
    gains achieved by the Anatomical Strip Head (ASR-HC).
    """
    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 40,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(
        self,
        feature_map: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            feature_map: Tensor in R^(B x 1024 x H x W)
            return_features: whether to return pooled feature vector
            
        Returns:
            logits: Tensor in R^(B x 40)
            pooled_feat (optional): Tensor in R^(B x 1024)
        """
        pooled = self.pool(feature_map).flatten(1)  # (B, C)
        normed = self.norm(pooled)
        dropped = self.dropout(normed)
        logits = self.classifier(dropped)
        
        if return_features:
            return logits, pooled
        return logits


# ==============================================================================
# NATIVE PURE-PYTORCH CONVNEXT-BASE (Air-Gapped & Zero-Dependency Fallback)
# ==============================================================================

class _NativeLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class _NativeMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Identity()
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class _NativeConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _NativeMlp(dim, 4 * dim)
        self.shortcut = nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + x
        return x

class _NativeConvNeXtStage(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, depth: int, stride: int = 2) -> None:
        super().__init__()
        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                _NativeLayerNorm2d(in_chs, eps=1e-6),
                nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=stride),
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[_NativeConvNeXtBlock(out_chs) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class NativeConvNeXtBase(nn.Module):
    """
    Pure PyTorch native ConvNeXt-Base backbone.
    Exact bit-for-bit, layer-for-layer match with timm convnext_base.
    Requires ZERO external libraries (no timm, no safetensors, no huggingface_hub).
    Guarantees air-gapped execution on CodaBench and offline servers.
    """
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=4),
            _NativeLayerNorm2d(128, eps=1e-6),
        )
        self.stages = nn.Sequential(
            _NativeConvNeXtStage(128, 128, depth=3, stride=1),
            _NativeConvNeXtStage(128, 256, depth=3, stride=2),
            _NativeConvNeXtStage(256, 512, depth=27, stride=2),
            _NativeConvNeXtStage(512, 1024, depth=3, stride=2),
        )
        self.norm_pre = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x


# ==============================================================================
# JEV MULTI-SPLIT SHARED-STEM INFERENCE ENGINE
# ==============================================================================

class JEVMultiSplitEngine(nn.Module):
    """
    Joint Execution Vectorization (JEV) Shared-Stem Inference Engine.
    
    Houses the shared ConvNeXt-Base backbone and coordinates multi-head / multi-split
    execution with zero redundant computation.
    
    Key Architectural Guarantees:
    1. Stages 0..2 Frozen:
       - Backbone Stem and Stages 0, 1, and 2 are strictly frozen (requires_grad = False).
       - Stage 4 (Stage index 3 in timm ConvNeXt) is trainable and fine-tuned for
         spatial anatomical feature extraction.
    2. Joint Execution Vectorization (JEV):
       - A single backbone forward pass extracts shared representations for
         both ASR-HC and Baseline heads simultaneously ("both" mode).
       - Enables vectorized multi-dataset split inference (e.g., Market1501, PA-100K,
         PETA) via batched joint tensor execution.
    """
    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = False,
        num_classes: int = 40,
        dropout: float = 0.1,
        in_channels: int = 1024,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Instantiate ConvNeXt-Base backbone (uses pure PyTorch NativeConvNeXtBase when offline/pretrained=False)
        if pretrained and _HAS_TIMM:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
            )
        else:
            self.backbone = NativeConvNeXtBase()
        
        # Strictly freeze ConvNeXt-Base Stem and Stages 0, 1, 2
        self._freeze_stages_0_to_2()
        
        # Dedicated Heads
        self.asr_head = AnatomicalStripHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.baseline_head = BaselineGAPHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _freeze_stages_0_to_2(self) -> None:
        """Freezes ConvNeXt-Base stem and stages 0, 1, 2 (stages 0..2)."""
        # Freeze stem
        if hasattr(self.backbone, "stem"):
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
                
        # Freeze stages 0, 1, 2
        if hasattr(self.backbone, "stages"):
            for stage_idx in range(min(3, len(self.backbone.stages))):
                for p in self.backbone.stages[stage_idx].parameters():
                    p.requires_grad = False

    def is_stage_frozen(self, stage_idx: int) -> bool:
        """Utility to verify frozen status of a given stage index."""
        if hasattr(self.backbone, "stages") and stage_idx < len(self.backbone.stages):
            return all(not p.requires_grad for p in self.backbone.stages[stage_idx].parameters())
        return False

    def get_frozen_status_report(self) -> Dict[str, bool]:
        """Returns a verification report of parameter freezing across backbone stages."""
        report = {}
        if hasattr(self.backbone, "stem"):
            report["stem_frozen"] = all(not p.requires_grad for p in self.backbone.stem.parameters())
        if hasattr(self.backbone, "stages"):
            for i in range(len(self.backbone.stages)):
                stage_name = f"stage_{i}_frozen"
                report[stage_name] = all(not p.requires_grad for p in self.backbone.stages[i].parameters())
        return report

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass through shared stem, stages 0..2 (frozen), and
        stage 4 (trainable) up to the pre-pooling norm.
        
        Returns:
            feature_map: Tensor in R^(B x 1024 x H x W)
        """
        # In timm ConvNeXt: forward_features = stem -> stages[0..3] -> norm_pre
        feature_map = self.backbone.forward_features(x)
        return feature_map

    def forward(
        self,
        x: torch.Tensor,
        head: str = "asr",
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any], Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Joint forward inference execution.
        
        Args:
            x: Input image tensor in R^(B x 3 x H x W)
            head: Head selection - "asr" (default ASR-HC), "baseline", or "both"
            return_details: Whether to return intermediate tokens/maps
            
        Returns:
            - If head == "asr": Logits tensor in R^(B x 40) [or (logits, details)]
            - If head == "baseline": Logits tensor in R^(B x 40)
            - If head == "both": Dict {"asr": asr_logits, "baseline": baseline_logits, ...}
        """
        # Shared stem & backbone feature extraction (single pass)
        feat_map = self.extract_features(x)
        
        if head == "asr":
            return self.asr_head(feat_map, return_details=return_details)
        elif head == "baseline":
            return self.baseline_head(feat_map)
        elif head == "both":
            asr_res = self.asr_head(feat_map, return_details=return_details)
            base_logits = self.baseline_head(feat_map)
            
            if return_details:
                asr_logits, details = asr_res
                return {
                    "asr": asr_logits,
                    "baseline": base_logits,
                    "features": feat_map,
                    "details": details,
                }
            return {
                "asr": asr_res,
                "baseline": base_logits,
                "features": feat_map,
            }
        else:
            raise ValueError(f"Unknown head type '{head}'. Expected 'asr', 'baseline', or 'both'.")

    def forward_joint_splits(
        self,
        split_tensors: Dict[str, torch.Tensor],
        head: str = "asr",
    ) -> Dict[str, torch.Tensor]:
        """
        Vectorized Multi-Split Execution:
        Concatenates batches from multiple dataset sources (e.g., Market1501, PA-100K, PETA)
        into a single contiguous tensor, executes joint backbone feature extraction,
        and unbinds predictions back into per-split outputs.
        
        Eliminates per-split scheduling overhead and maximizes GPU/CPU compute density.
        """
        split_names = list(split_tensors.keys())
        batch_sizes = [split_tensors[k].shape[0] for k in split_names]
        
        # Contiguous concatenation
        x_joint = torch.cat([split_tensors[k] for k in split_names], dim=0)
        
        # Single vectorized joint execution
        joint_logits = self.forward(x_joint, head=head, return_details=False)
        
        # Split outputs back to source keys
        out = {}
        curr = 0
        for name, bs in zip(split_names, batch_sizes):
            out[name] = joint_logits[curr : curr + bs]
            curr += bs
        return out


def create_jev_asrhc_model(pretrained: bool = False, **kwargs) -> JEVMultiSplitEngine:
    """Helper factory function to instantiate the JEV ASR-HC architecture."""
    return JEVMultiSplitEngine(pretrained=pretrained, **kwargs)
