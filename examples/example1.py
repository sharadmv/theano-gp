import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from gp import GaussianProcess
from gp.kernel import RBF

if __name__ == "__main__":
    kernel = RBF(1.0)
    gp = GaussianProcess(kernel).compile()

    x = np.arange(-5, 5, 0.1)
    plt.figure()
    num_samples = 10
    samples = gp.sample(x[:, np.newaxis], num_samples)
    for i in xrange(num_samples):
        plt.plot(x, samples[i])
    plt.show()

    gp.observe([[1], [2]], [0, 1])
    plt.figure()
    num_samples = 10
    samples = gp.sample(x[:, np.newaxis], num_samples)
    for i in xrange(num_samples):
        plt.plot(x, samples[i])
    plt.show()
