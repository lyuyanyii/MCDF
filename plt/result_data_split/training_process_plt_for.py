import matplotlib.pyplot as plt
import numpy as np

alphas = [0, 0.125, 0.5, 1]

accs = []
for alpha in alphas:
    with open("../../result/cifar10sub_seed1/alpha_{}_percent_100/log.log".format(alpha), 'r') as f:
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
        accs.append( int((1-acc)*10000)/100 )

x = range(len(accs))
width = 1/1.5
plt.xlabel('Alpha')
plt.ylabel('Error rate')
plt.bar( x, accs, width, tick_label = [0, 0.125, 0.5, 1] )
for i, v in enumerate(accs):
    plt.text( i-0.1, v+0.05, str(v) , color='red')
plt.savefig( 'seed_1.png' )
plt.show()
"""
plt.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig( 'val_d124.png' )
"""
