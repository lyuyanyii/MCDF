import torch.nn as nn
import torch
import numpy as np

class A( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(1))

    def forward( self, x ):
        print(self.running_mean)
        self.running_mean[0] += 1
        return x

model = torch.nn.DataParallel( A() ).cuda()
#model = A()

x = model( torch.zeros(10).cuda() )

if hasattr( model, 'module' ):
    print( model.module.running_mean )
else:
    print( model.running_mean )
