import torch
import torch.nn as nn
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import SegformerForSemanticSegmentation,SegformerConfig


class SegFormerB0_12Channel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        config = SegformerConfig(
        num_labels=num_classes,
        hidden_sizes=[32, 64, 160, 256],
        num_attention_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        decoder_hidden_size=256,
        )

        self.model = SegformerForSemanticSegmentation(config)
        original_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=12,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )
        return logits_upscaled