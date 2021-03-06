import matplotlib.pyplot as plt
import pickle
import numpy as np

alphas = [0.125, 0.25, 0.5, 1, 2, 4, 8]
percents = [20]
accs = []

for per in percents:
    acc = []
    for alpha in alphas:
        with open("../result/cifar10sub/alpha_{}_percent_{}/log.log".format(alpha, per), "r") as f:
            lines = f.readlines()
            #line = lines[-2]
            #acc.append( float(line.split(" ")[-1]) )
            s = []
            for line in lines:
                if 'VAL FINAL' in line:
                    s.append( float(line.split(" ")[-1]) )
            s = s[-10:-1]
            acc.append( 1-np.array(s).mean() )

    accs.append(acc)
    plt.plot( alphas, acc, label="MCDF" )

    with open("../result/cifar10sub/alpha_0_percent_{}/log.log".format(per), "r") as f:
        lines = f.readlines()
        #line = lines[-2]
        #acc.append( float(line.split(" ")[-1]) )
        s = []
        for line in lines:
            if 'VAL FINAL' in line:
                s.append( float(line.split(" ")[-1]) )
        s = s[-10:-1]
        acc = [1-np.array(s).mean()] * len(alphas)
    plt.plot( alphas, acc, label='baseline'.format(per) )


"""
plt.plot( list(np.array(range(T+1, 11)) * 10), CAM, label = 'CAM' )
#plt.plot( list(np.array(range(T+1, 11)) * 10), Mask, label = 'Mask' )
#plt.plot( list(np.array(range(T+1, 11)) * 10), Mask_2Reg, label = 'Mask_2Reg' )
plt.plot( list(np.array(range(T+1, 11)) * 10), Mask_CAM, label = 'Ours+CAM' )
plt.plot( list(np.array(range(1, 11)) * 10), Random, label = 'Random' )
plt.plot( list(np.array(range(T+1, 11)) * 10), Mask_KL_reg, label = 'Ours' )
"""

plt.xscale('log')
plt.grid(True)
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.legend()
plt.title("CIFAR10 with 20 Percentage of Training Data")
plt.savefig( '20_percent_cifar10.pdf', format='pdf' )
plt.show()
#plt.savefig( 'Figure1.pdf', format='pdf' )
