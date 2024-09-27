import torch
from torch import nn
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model

@register_feature_model(name="sdvae")
class SDVAE(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        self.model = AutoencoderKL.from_pretrained(self.config.path, subfolder="vae")
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512)#, interpolation=transforms.InterpolationMode.BICUBIC
                ),
            ]
        )

    def compute_latent(self, x):
        x = self.preprocess(x)
        y = self.model.encode(x).latent_dist.sample()
        return y.flatten(start_dim=1)
