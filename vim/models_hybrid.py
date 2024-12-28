import torch
import torch.nn as nn
from timm import create_model
from efficientnet_pytorch import EfficientNet
from VisionMamba import VisionMamba

class HybridMambaEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridMambaEfficientNet, self).__init__()
        # Mamba model (primary backbone)
        self.mamba = VisionMamba(
            patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean',
            if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
            if_cls_token=True, if_divide_out=True, use_middle_cls_token=True,
            num_classes=0  # No classification head in Mamba
        )

        # EfficientNet (secondary feature extractor)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        efficient_in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove classification head

        # Fusion layer
        print(f"Mamba: {self.mamba.num_features}, Eff: {efficient_in_features}, Combined: {self.mamba.num_features + efficient_in_features}")
        self.fusion_layer = nn.Linear(self.mamba.num_features + efficient_in_features, 512)

        # print(self.efficientnet)
        # print()
        # print(self.mamba)

        # Classification head
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Mamba features
        mamba_features = self.mamba(x)

        # EfficientNet features
        eff_features = self.efficientnet(x)
        eff_features = torch.flatten(eff_features, start_dim=1)

        # Fuse features
        combined_features = torch.cat([mamba_features, eff_features], dim=1)
        # print(f"Mamba: {mamba_features.shape}, Eff: {eff_features.shape}, Combined: {combined_features.shape}")
        fused_features = self.fusion_layer(combined_features)
        fused_features = nn.ReLU()(fused_features)

        # Final classification
        out = self.classifier(fused_features)
        return out

