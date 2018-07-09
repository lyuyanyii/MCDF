import matplotlib.pyplot as plt
import numpy as np

alphas = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8]
alphas = [0, 0.125, 1]

for alpha in alphas:
    with open("../result/cifar10sub/alpha_{}_percent_100/log.log".format(alpha), 'r') as f:
        lines = f.readlines()
        s = []
        for line in lines:
            if 'VAL FINAL' in line:
                s.append( float(line.split(" ")[-1]) )
        T = 1
        s = np.convolve( np.ones(T)/T, np.array(s), mode='valid' )
        plt.plot( range(len(s)), s, label='alp={}'.format(alpha) )

plt.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig( 'val_d124.png' )
