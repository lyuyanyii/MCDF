import matplotlib.pyplot as plt
import numpy as np

alphas = [0, 0.125, 0.5, 1]

accs = []
for alpha in alphas:
    with open("../../result/cifar10sub/alpha_{}_percent_100/log.log".format(alpha), 'r') as f:
        lines = f.readlines()
        s = []
        for line in lines:
            if 'VAL FINAL' in line:
                s.append( float(line.split(" ")[-1]) )
        """
        T = 1
        s = np.convolve( np.ones(T)/T, np.array(s), mode='valid' )
        plt.plot( range(len(s)), s, label='alp={}'.format(alpha) )
        """
        acc = np.array(s[-10: -1]).mean()
        accs.append( (1-acc) )

x = range(len(accs))
plt.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Error')
plt.title( 'Cifar10_split2' )
plt.plot( alphas, accs, label='D124_K12' )
#for i, v in enumerate(accs):
#    plt.text( i-0.1, v+0.05, str(v) , color='red')
plt.savefig( 'split_defualt.pdf' )
plt.show()
"""
plt.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig( 'val_d124.png' )
"""
