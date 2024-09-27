# based on https://github.com/sthalles/PyTorch-BYOL/
import torchvision.models as models
import torch
from torch import nn
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model
from torchvision import transforms
import torch
import yaml
import os


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class ResNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18()
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50()

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)


@register_feature_model(name="byol")
class BYOL(BaseFeatureModel, nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.config = model_config
        base_dir = os.path.dirname(__file__)
        device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(base_dir, model_config.model_path)
        self.model = models.resnet50()
        self.model.load_state_dict(torch.load(model_path))

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512)#, interpolation=transforms.InterpolationMode.BICUBIC
                ),
            ]
        )

        self.model = self.model.to(device)

        self.representation = nn.Sequential(*list(self.model.children())[:-1])


    def compute_latent(self, x):
        # B x C x H x W -- 0 to 1 
        x_in = self.preprocess(x)
        with torch.no_grad(): 
            y = self.representation(x_in).flatten(start_dim=1)
        return y
