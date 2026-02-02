import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3Small(nn.Module):

    def __init__(self, num_classes=11):
        super().__init__()

        self.model = mobilenet_v3_small(weights=None)
        state_dict = torch.load('.\\mobilenet_v3_small-047dcff4.pth')
        self.model.load_state_dict(state_dict)
        self.model.features[0][0] = nn.Conv2d(
            in_channels=12,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.model.features[0][1] = nn.BatchNorm2d(16)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
   
    def forward(self,x):
        outputs = self.model(x)
        return outputs
