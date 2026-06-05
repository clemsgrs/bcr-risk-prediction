"""
HIPT models with DCTM survival head.

These models combine the HIPT backbone (without the final linear classifier)
with a DCTM (Deep Conditional Transformation Model) head for continuous-time
survival analysis.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from DCTM.dctm import DCTM_general_shift, DCTM_general_shift_scale, DCTM_general

from hipt.src.models.vision_transformer import vit4k_xs
from hipt.src.models.components import Attn_Net_Gated
from hipt.src.models.utils import update_state_dict
from bcr.train.dctm_eval import dctm_survival_from_transform


class LocalHIPTWithDCTM(nn.Module):
    """
    Local HIPT backbone with DCTM survival head.

    Takes pre-extracted patch-level features, processes them through a region-level
    vision transformer, then aggregates to slide-level and produces survival predictions
    using DCTM's continuous-time hazard parameterization.

    Args:
        region_size: Size of region in pixels (default: 2048)
        patch_size: Size of patch in pixels (default: 256)
        embed_dim_patch: Dimension of patch features (default: 384)
        embed_dim_region: Dimension of region features (default: 192)
        embed_dim_slide: Dimension of slide-level embedding (default: 192)
        dropout: Dropout rate (default: 0.25)
        mask_attn: Whether to use masked attention (default: False)
        num_register_tokens: Number of register tokens (default: 0)
        num_heads: Number of attention heads (default: 6)
        pretrained_weights: Path to pretrained weights (default: None)
        img_size_pretrained: Image size used for pretraining (default: None)
        dctm_variant: DCTM variant - 'shift' (DCTM^S), 'shift_scale' (DCTM^SS), or 'general' (DCTM^G)
        basis_features: Number of Bernstein basis features for DCTM (default: 6)
        family: DCTM distribution family, 'logistic' or 'gompertz' (default: 'logistic')
    """

    def __init__(
        self,
        region_size: int = 2048,
        patch_size: int = 256,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        dropout: float = 0.25,
        mask_attn: bool = False,
        num_register_tokens: int = 0,
        num_heads: int = 6,
        pretrained_weights: str | None = None,
        img_size_pretrained: int | None = None,
        dctm_variant: str = "shift",
        basis_features: int = 6,
        family: str = "logistic",
    ):
        super(LocalHIPTWithDCTM, self).__init__()
        self.npatch = int(region_size // patch_size)
        self.num_register_tokens = num_register_tokens
        self.embed_dim_slide = embed_dim_slide
        self.family = family

        checkpoint_key = "teacher"

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
            mask_attn=mask_attn,
            img_size_pretrained=img_size_pretrained,
            num_register_tokens=num_register_tokens,
            num_heads=num_heads,
        )

        if pretrained_weights and Path(pretrained_weights).is_file():
            print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                model_dict=self.vit_region.state_dict(), state_dict=state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=True)
            print(f"Pretrained weights found at {pretrained_weights}")
            print(msg)

        elif pretrained_weights:
            print(
                f"{pretrained_weights} doesnt exist ; please provide path to existing file"
            )

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_slide,
                nhead=3,
                dim_feedforward=embed_dim_slide,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=embed_dim_slide, D=embed_dim_slide, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(embed_dim_slide, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)]
        )

        # DCTM survival head - select variant
        dctm_classes = {
            "shift": DCTM_general_shift,
            "shift_scale": DCTM_general_shift_scale,
            "general": DCTM_general,
        }
        if dctm_variant not in dctm_classes:
            raise ValueError(
                f"Unknown DCTM variant '{dctm_variant}'. "
                f"Must be one of: {list(dctm_classes.keys())}"
            )
        self.dctm_head = dctm_classes[dctm_variant](
            input_features=embed_dim_slide,
            basis_features=basis_features,
            family=family,
        )

    def get_embedding(
        self, x: torch.Tensor, pct: torch.Tensor | None = None, pct_thresh: float = 0.0
    ) -> torch.Tensor:
        """
        Extract slide-level embedding from patch features.

        Args:
            x: Patch features of shape [1, num_regions, num_patches, embed_dim_patch]
            pct: Optional percentage tensor for masking
            pct_thresh: Threshold for masking

        Returns:
            Slide embedding of shape [1, embed_dim_slide]
        """
        mask_patch = None
        if pct is not None:
            pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
            mask_patch = (pct_patch > pct_thresh).int()  # (M, npatch**2)
            # add the [CLS] token to the mask
            cls_token = mask_patch.new_ones((mask_patch.size(0), 1))
            # eventually add register tokens to the mask
            if self.num_register_tokens:
                register_tokens = mask_patch.new_ones(
                    (mask_patch.size(0), self.num_register_tokens)
                )
                mask_patch = torch.cat((cls_token, register_tokens, mask_patch), dim=1)
            else:
                mask_patch = torch.cat((cls_token, mask_patch), dim=1)

        # x = [1, num_regions, num_patches, embed_dim_patch]
        x = x.squeeze(0)
        x = self.vit_region(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2),
            mask=mask_patch,
        )  # [num_regions, embed_dim_region]
        x = self.global_phi(x)  # [num_regions, embed_dim_slide]

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        return x_wsi  # [1, embed_dim_slide]

    def forward(
        self,
        x: torch.Tensor,
        time: np.ndarray,
        pct: torch.Tensor | None = None,
        pct_thresh: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass returning the DCTM transformation value h(t|x).

        Args:
            x: Patch features of shape [1, num_regions, num_patches, embed_dim_patch]
            time: Normalized event times as numpy array in [0, 1]
            pct: Optional percentage tensor for masking
            pct_thresh: Threshold for masking

        Returns:
            DCTM transformation value of shape [batch_size]
        """
        embedding = self.get_embedding(x, pct=pct, pct_thresh=pct_thresh)
        return self.dctm_head(embedding, time)

    def predict_transform(
        self,
        x: torch.Tensor,
        time: np.ndarray,
        pct: torch.Tensor | None = None,
        pct_thresh: float = 0.0,
    ) -> torch.Tensor:
        """Predict DCTM transformation values h(t|x) at normalized times."""
        return self.forward(x, time, pct=pct, pct_thresh=pct_thresh)

    def predict_survival(
        self,
        x: torch.Tensor,
        time: np.ndarray,
        pct: torch.Tensor | None = None,
        pct_thresh: float = 0.0,
    ) -> torch.Tensor:
        """Predict survival probabilities S(t|x) at normalized times."""
        transform = self.predict_transform(x, time, pct=pct, pct_thresh=pct_thresh)
        return dctm_survival_from_transform(transform, self.family)

    def compute_loss(
        self,
        x: torch.Tensor,
        time: np.ndarray,
        event: torch.Tensor,
        pct: torch.Tensor | None = None,
        pct_thresh: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute DCTM negative log-likelihood loss.

        Args:
            x: Patch features of shape [1, num_regions, num_patches, embed_dim_patch]
            time: Normalized event times as numpy array in [0, 1]
            event: Event indicator (1=event occurred, 0=censored)
            pct: Optional percentage tensor for masking
            pct_thresh: Threshold for masking

        Returns:
            Scalar loss value
        """
        embedding = self.get_embedding(x, pct=pct, pct_thresh=pct_thresh)
        return self.dctm_head.nllloss(embedding, time, event)

    def predict_tte(
        self, x: torch.Tensor, pct: torch.Tensor | None = None, pct_thresh: float = 0.0
    ) -> np.ndarray:
        """
        Predict time-to-event.

        Args:
            x: Patch features of shape [1, num_regions, num_patches, embed_dim_patch]
            pct: Optional percentage tensor for masking
            pct_thresh: Threshold for masking

        Returns:
            Predicted time-to-event as numpy array (normalized scale [0, 1])
        """
        embedding = self.get_embedding(x, pct=pct, pct_thresh=pct_thresh)
        return self.dctm_head.tte(embedding)
