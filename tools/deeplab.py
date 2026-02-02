import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        ))
        
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                         padding=rate, dilation=rate, bias=False),
                nn.ReLU(inplace=True)
            ))
        
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], 
                               mode='bilinear', align_corners=True)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusMobileNetV3(nn.Module):
    def __init__(self, num_classes, pretrained_backbone_path=None):
        super().__init__()
        
        backbone = models.mobilenet_v3_small(weights = None)
        first_conv = backbone.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=12,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        backbone.features[0][0] = new_first_conv
        if pretrained_backbone_path is not None:
            pretrained_dict = torch.load(pretrained_backbone_path, map_location='cpu')
            model_dict = backbone.state_dict()
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and not k.startswith('features.0.0')
            }
            model_dict.update(filtered_dict)
            backbone.load_state_dict(model_dict)

        features = backbone.features
        self.low_level_features = nn.Sequential(*features[:4])
        self.high_level_features = nn.Sequential(*features[4:])
        self.aspp = ASPP(in_channels=576, atrous_rates=[6, 12, 18])
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        low_feat = self.low_level_features(x)
        high_feat = self.high_level_features(low_feat)

        x = self.aspp(high_feat)
        x = nn.functional.interpolate(x, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)

        low_feat = self.low_level_conv(low_feat)
        x = torch.cat([x, low_feat], dim=1)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x
