import matplotlib.pyplot as plt
import numpy as np

alphas = [0, 0.125, 0.5, 1]
percents = [100]

for per in percents:
    acc = []
    for alpha in alphas:
        with open("../../result/cifar100sub_seed0/alpha_{}_percent_{}_Gaussian/log.log".format(alpha, per), "r") as f:
            lines = f.readlines()
            s = []
            for line in lines:
                if 'VAL FINAL' in line:
                    s.append( float(line.split(" ")[-1]) )
            s = s[-10:-1]
            acc.append( 1 - np.array(s).mean() )
    plt.plot( alphas, acc, label='D124_K12' )

plt.grid(True)
plt.xlabel( 'Alpha' )
plt.ylabel( 'Error' )
plt.title( 'Cifar100' )
plt.savefig( 'cifar100.pdf', format='pdf' )
plt.show()
