import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

def plot(fx,x,title):
    plt.figure(1)
    plt.title(title)
    plt.plot(fx, color='b', label = 'Energy')
    plt.xlabel('nodes')
    plt.ylabel('energy')
    plt.legend(loc='best')
    plt.savefig('{:4d}_string.png'.format(title),dpi=600)


