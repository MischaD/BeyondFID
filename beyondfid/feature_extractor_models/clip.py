import torch
from torch import nn
import torchvision.transforms as transforms
import clip
from . import BaseFeatureModel, register_feature_model


@register_feature_model(name="clip")
class CLIP(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.features_size = 512
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    224
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model.eval()

    def compute_latent(self, x):
        x = self.preprocess(x)
        with torch.no_grad():
            y = self.model.encode_image(x)
        return y


