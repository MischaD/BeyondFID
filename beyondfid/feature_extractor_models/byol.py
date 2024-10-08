# based on https://github.com/sthalles/PyTorch-BYOL/
import torchvision.models as models
import torch
from torch import nn
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model
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
        config = yaml.load(open(os.path.join(model_config.cfg_path), "r"), Loader=yaml.FullLoader)

        device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        encoder = ResNet(**config['network'])

        #load pre-trained parameters
        load_params = torch.load(os.path.join(os.path.join(model_config.model_path)),
                                map_location=torch.device(torch.device(device)))

        if 'online_network_state_dict' in load_params:
            encoder.load_state_dict(load_params['online_network_state_dict'])
        self.output_feature_dim = encoder.projetion.net[0].in_features

        # remove the projection head
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
        self.encoder = encoder.to(device)


    def compute_latent(self, x):
        # B x C x H x W -- 0 to 1 
        with torch.no_grad(): 
            y = self.encoder(x).squeeze()
        return y

