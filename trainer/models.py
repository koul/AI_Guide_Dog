import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import *

class ResNet18(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, fixconvs=False, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, 3)
        self.dropout = torch.nn.Dropout(p=0.07)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # model.fc.weight.requires_grad = True
        # model.fc.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
        x = self.dropout(x)
        x = self.regressor(x)
        return x