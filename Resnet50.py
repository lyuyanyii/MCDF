import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized, ThresholdBinarized, sharp_t
from torchvision import models
import torch.utils.model_zoo as model_zoo
import utils

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
	}

class Cls( nn.Module ):
    def __init__( self, pretrained = True ):
        super().__init__()
        resnet = models.resnet50()
        if pretrained:
            resnet.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.conv1 = nn.Sequential( resnet.conv1, resnet.bn1, resnet.relu )
        self.layer0 = nn.Sequential( resnet.maxpool, resnet.layer1 )
        self.layer1, self.layer2, self.layer3 = resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.ndf = 2048
        if not pretrained:
            self.apply( weight_init )

    def forward( self, x ):
        x0 = self.conv1( x )
        x1 = self.layer0( x0 )
        x2 = self.layer1( x1 )
        x3 = self.layer2( x2 )
        x4 = self.layer3( x3 )
        #f = self.avgpool( x4 )
        f = x4.mean(3).mean(2)
        f = f.view( f.size(0), -1 )
        pred = self.fc( f )
        return pred, f

