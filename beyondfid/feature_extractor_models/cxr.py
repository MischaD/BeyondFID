import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as T 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from beyondfid.utils import ToTensorIfNotTensor

from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model

@register_feature_model(name="cxr")
class CXR(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = get_classification_model(model_config.path, model_config.n_classes)
        # transform? input is B x 3 x H x W from 0 to 1 (RBG)
        self.setup = False

    def do_setup(self, x): 
        self.setup = True
        self.model = self.model.to(x.device)
        self.model.eval()

    def compute_latent(self, x):
        if not self.setup: 
            self.do_setup(x)
            
        y = self.model(x)
        return y


def get_classification_model(model_path, n_classes): 

    class DenseNet121(nn.Module):

        def __init__(self, classCount, isTrained):
        
            super(DenseNet121, self).__init__()
            
            self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

            kernelCount = self.densenet121.classifier.in_features
            
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

        def forward(self, x):
            x = self.densenet121(x)
            return x

    cudnn.benchmark = True
    
    #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
    model = DenseNet121(n_classes, True)
    model = model

    modelCheckpoint = torch.load(model_path)
    state_dict = {k[7:]:v for k, v in modelCheckpoint['state_dict'].items()}
    model.load_state_dict(state_dict)


    class Classifier(nn.Module): 
        def __init__(self, model, transforms="default") -> None:
            super().__init__()
            if transforms == "default": 
                normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                transformList = []
                #transformList.append(T.Resize(256)) -- forward pass during inference uses tencrop 
                transformList.append(T.Resize(226))
                transformList.append(T.CenterCrop(226))
                transformList.append(ToTensorIfNotTensor())
                transformList.append(normalize)
                self.transforms=T.Compose(transformList)
            else: 
                self.transforms = transforms
            self.model = model

        def forward(self, x): 
            x_in = self.transforms(x)
            return self.model(x_in)
        
        def lazy_foward(self, x): 
            # accepts tensor, 0-1, bchw 
            self.model.eval()
            self.model.to("cuda")
            
            with torch.no_grad():
                x_in = self.transforms(x)
                if x_in.dim() == 3: 
                    x_in = x_in.unsqueeze(dim=0)
                
                varInput = x_in.cuda()

                features = self.model.densenet121.features(varInput)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                hidden_features = torch.flatten(out, 1)
                out = self.model.densenet121.classifier(hidden_features)
                #outMean = out.view(bs, ).mean(1)
            return out.data, hidden_features.data

    return Classifier(model)
