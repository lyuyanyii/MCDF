import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init, Binarized

class DenseLayer(nn.Module):
    def __init__(self, inp_chl, growth_rate, bn_size = 4):
        super().__init__()
        self.bn1   = nn.BatchNorm2d( inp_chl )
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d( inp_chl, bn_size * growth_rate, 1, stride = 1, bias = False )
        self.bn2   = nn.BatchNorm2d( bn_size * growth_rate )
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d( bn_size * growth_rate, growth_rate, 3, stride = 1, padding = 1, bias = False )

    def forward( self, x ):
        y = x
        y = self.bn1  ( y )
        y = self.relu1( y )
        y = self.conv1( y )
        y = self.bn2  ( y )
        y = self.relu2( y )
        y = self.conv2( y )
        return torch.cat([x, y], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, inp_chl, growth_rate, bn_size = 4):
        super().__init__()
        self.layers = nn.Sequential( *[ DenseLayer( inp_chl + i*growth_rate, growth_rate, bn_size ) for i in range(num_layers) ] )

    def forward( self, x ):
        x = self.layers( x )
        return x

class Transition(nn.Module):
    def __init__(self, inp_chl, out_chl):
        super().__init__()
        self.bn = nn.BatchNorm2d( inp_chl, out_chl )
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d( inp_chl, out_chl, 1, stride = 1, bias = False )
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward( self, x ):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class Cls(nn.Module):
    def __init__( self, growth_rate = 12, layers = [10, 10, 10], bn_size = 4, reduce_dim=False, ncls=10 ):
        super().__init__()

        self.conv1 = nn.Conv2d( 3, growth_rate * 2, 3, stride = 1, padding = 1 )
        self.encoder1 = DenseBlock( layers[0], growth_rate * 2, growth_rate, bn_size )
        self.chl1 = growth_rate * (2 + layers[0])
        self.t1 = Transition( self.chl1, self.chl1 )
        self.encoder2 = DenseBlock( layers[1], self.chl1, growth_rate, bn_size )
        self.chl2 = self.chl1 + growth_rate * layers[1]
        self.t2 = Transition( self.chl2, self.chl2 )
        self.encoder3 = DenseBlock( layers[2], self.chl2, growth_rate, bn_size )
        self.chl3 = self.chl2 + growth_rate * layers[2]

        self.reduce_dim = reduce_dim
        if not reduce_dim:
            self.fc = nn.Linear( self.chl3, ncls )
        else:
            self.fc1 = nn.Sequential(
                nn.ReLU(True), 
                nn.Linear( self.chl3, 64 ),
                )
            self.fc = nn.Linear( 64, ncls )

        self.apply( weight_init )

        if not reduce_dim:
            self.ndf = self.chl3
        else:
            self.ndf = 64

    def forward( self, x ):
        x = self.conv1( x )
        x0 = self.encoder1( x )
        x1 = self.encoder2( self.t1( x0 ) )
        x2 = self.encoder3( self.t2( x1 ) )
        f = x2.mean(3).mean(2)
        if not self.reduce_dim:
            pred = self.fc( f )
        else:
            f = self.fc1( f )
            pred = self.fc( f )
        return pred, f

