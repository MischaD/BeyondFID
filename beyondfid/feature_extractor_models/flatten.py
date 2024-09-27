import torch
from torch import nn
import torchvision.transforms as transforms
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model


@register_feature_model(name="flatten")
class Flatten(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

    def compute_latent(self, x):
        return x.flatten(start_dim=1)


@register_feature_model(name="flatten_resize")
class FlattenResize(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.size = model_config.resize_to

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (self.size, self.size)
                ),
            ]
        )

    def compute_latent(self, x):
        x = self.preprocess(x)
        y = x.flatten(start_dim=1)
        return y