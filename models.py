import Resnet20 
import resnet
import BNet
import VGG_CIFAR
import Resnet50

def Res20(**kwargs):
    return Resnet20.Net()

def Res110( sto_depth, **kwargs ):
    if sto_depth:
        mode = None
    else:
        mode = 'linear'
    return resnet.createModel( 110, 'cifar10', 10, death_mode=mode )

def Densenet( **kwargs ):
    return BNet.Cls()

def Densenet124( reduce_dim=False, ncls=10, **kwargs ):
    return BNet.Cls( layers=[10,20,30], reduce_dim=reduce_dim, ncls=ncls )

def VGG( **kwargs ):
    return VGG_CIFAR.Cls()

def Res50( pretrained, **kwargs ):
    return Resnet50.Cls( pretrained=pretrained )
