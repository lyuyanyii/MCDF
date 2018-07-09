import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized

class ResLayer(nn.Module):
    def __init__(self, inp_chl, output_chl, ker_size = 3, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d( inp_chl, output_chl, ker_size, padding = (ker_size - 1)//2, stride=stride )
        self.bn1 = nn.BatchNorm2d( output_chl )
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d( output_chl, output_chl, ker_size, padding = (ker_size-1)//2 , stride = 1 )
        self.bn2 = nn.BatchNorm2d( output_chl )
        self.relu2 = nn.ReLU(True)
        if stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d( inp_chl, output_chl, 1, stride = stride ),
                nn.BatchNorm2d( output_chl ),
                )
        else:
            self.proj = None

    def forward( self, x ):
        inp = x
        x = self.conv1( x )
        x = self.bn1( x )
        x = self.relu1( x )
        x = self.conv2( x )
        x = self.bn2( x )
        if self.proj is not None:
            inp = self.proj( inp )
        x += inp
        x = self.relu2( x )
        return x

class Net( nn.Module ):
    def __init__( self ):
        super().__init__()
        chls = [16, 32, 64]
        k = 18
        self.conv0 = nn.Sequential(
            nn.Conv2d( 3, chls[0], 3, padding = 1 ),
            )
        self.layer1 = nn.Sequential(
            *[ ResLayer( chls[0], chls[0] ) for _ in range(k) ]
            )
        self.layer2 = nn.Sequential(
            ResLayer( chls[0], chls[1], stride = 2 ),
            *[ ResLayer( chls[1], chls[1] ) for _ in range(k - 1) ]
            )
        self.layer3 = nn.Sequential(
            ResLayer( chls[1], chls[2], stride = 2 ),
            *[ ResLayer( chls[2], chls[2] ) for _ in range(k - 1) ]
            )
        self.fc = nn.Linear( chls[-1], 10 )
        
        self.apply( weight_init )

        self.ndf = chls[-1]

    def forward( self, x ):
        x = self.conv0( x )
        x = self.layer1( x )
        x = self.layer2( x )
        x = self.layer3( x )
        x = x.mean(3).mean(2)

        pred = self.fc( x )
        return pred, x
