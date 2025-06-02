import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool):
        super(EfficientNetGen, self).__init__()

        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained(model)
        else:
            self.efficientnet = EfficientNet.from_name(model)

        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, n_classes)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetB4(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained)
