import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized, ThresholdBinarized, sharp_t
from torchvision import models
import torch.utils.model_zoo as model_zoo
import utils

def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

class Cls( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv( 3, 64 ),
            conv( 64, 64 ),
            conv( 64, 128 ),
            conv( 128, 128 ),
            conv( 128, 256 ),
            conv( 256, 256 ),
            conv( 256, 512, stride = 2 ))
        self.layer0 = nn.Sequential(
            conv( 512, 512 ),
            conv( 512, 512 ),
            conv( 512, 512, stride = 2))
        self.layer1 = nn.Sequential(
            conv( 512, 512 ),
            conv( 512, 512 ),
            conv( 512, 512, stride = 2))
        self.layer2 = conv( 512, 512, stride = 2 )
        self.layer3 = conv( 512, 512, stride = 2 )
        self.fc1 = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.ReLU(True),
            )
        self.fc = nn.Linear( 512, 10 )
        self.pre_chls = [512, 512, 512, 512, 512]
        self.apply( weight_init )

        self.ndf = 512

    def forward( self, x ):
        x0 = self.conv1( x )
        x1 = self.layer0( x0 )
        x2 = self.layer1( x1 )
        x3 = self.layer2( x2 )
        x4 = self.layer3( x3 )
        #f = self.avgpool( x4 )
        f = x4.mean(3).mean(2)
        f = f.view( f.size(0), -1 )
        f = self.fc1( f )
        pred = self.fc( f )
        return pred, f

