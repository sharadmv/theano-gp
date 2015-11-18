import numpy as np
from theanify import Theanifiable, theanify
import theano.tensor as T

class KernelFunction(Theanifiable):

    def __init__(self):
        super(KernelFunction, self).__init__()

    def distance(self, X1, X2):
        raise NotImplementedError

    @theanify(T.matrix('X1'), T.matrix('X2'))
    def euclidean_distance(self, X1, X2):

        m = -(T.dot(X1, 2*X2.T)
                - T.sum(X1*X1, axis=1)[:,np.newaxis]
                - T.sum(X2*X2, axis=1)[:,np.newaxis].T)
        res = m * (m > 0.0)
        return res


class RBF(KernelFunction):

    def __init__(self, length_scale):
        super(RBF, self).__init__()
        self.length_scale = length_scale

    @theanify(T.matrix('X1'), T.matrix('X2'))
    def distance(self, X1, X2):
        ed2 = self.euclidean_distance(X1, X2)
        return T.exp(-ed2 / self.length_scale)
