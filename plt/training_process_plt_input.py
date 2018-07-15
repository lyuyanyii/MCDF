import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser( description='PLT' )
for i in range(10):
    parser.add_argument( '--plt{}'.format(i), type=str, metavar='PATH' )
    parser.add_argument( '--name{}'.format(i), type=str, help='name' )
parser.add_argument( '--save', type=str )
parser.add_argument( '--smooth', type=int, default=10 )
parser.add_argument( '--title', type=str )

args = parser.parse_args()
for i in range(10):
    dir = getattr( args, 'plt{}'.format(i) )
    name = getattr( args, 'name{}'.format(i) )
    if dir is None:
        continue
    with open(dir, 'r') as f:
        lines = f.readlines()
        s = []
        for line in lines:
            if 'VAL FINAL' in line:
                s.append( 1-float(line.split(" ")[-1]) )
            if 'Dims' in line:
                s = []
        T = args.smooth
        s = np.convolve( np.ones(T)/T, np.array(s), mode='valid' )
        plt.plot( range(len(s)), s, label=name )

plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title( args.title )
plt.legend()
if args.save is not None:
    plt.savefig( args.save, format='pdf' )
plt.show()
plt.close()
