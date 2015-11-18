import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theanify import theanify, Theanifiable

theano.config.on_unused_input = 'ignore'
from kernel import RBF

class GaussianProcess(Theanifiable):

    def __init__(self, kernel):
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
        self.rng = RandomStreams()
        self._observed = False

        self.Xtrain = theano.shared(np.zeros((0, 0)))
        self.ytrain = theano.shared(np.zeros(0))
        self.Kxx = theano.shared(np.zeros((0, 0)))
        self.Kxxy = theano.shared(np.zeros((0)))

    @theanify(T.vector('mean'), T.matrix('cov'), T.iscalar('size'))
    def mvg(self, mean, cov, size):
        D = mean.shape[0]
        u, s, v = T.nlinalg.svd(cov)
        x = self.rng.normal(size=(size, D))
        return T.dot(x, T.sqrt(s)[:, None] * v) + mean

    def mean(self, X):
        if self._observed:
            return self._mean_predictive(X)
        else:
            return self._mean_prior(X)

    def cov(self, X):
        if self._observed:
            return self._cov_predictive(X)
        else:
            return self._cov_prior(X)

    @theanify(T.matrix('X'))
    def _mean_prior(self, X):
        return T.zeros((X.shape[0],))

    @theanify(T.matrix('X'))
    def _mean_predictive(self, X):
        Kx_x = self.kernel.distance(X, self.Xtrain)
        return T.dot(Kx_x, self.Kxxy)

    @theanify(T.matrix('X'))
    def _cov_prior(self, X):
        return self.kernel.distance(X, X)

    @theanify(T.matrix('X'))
    def _cov_predictive(self, X):
        Kx_x = self.kernel.distance(X, self.Xtrain)
        Kx_x_ = self.kernel.distance(X, X)
        return Kx_x_ - T.dot(Kx_x, T.dot(self.Kxx, Kx_x.T))

    def observe(self, X, y):
        self._observed = True
        self._observe(X, y)

    @theanify(T.matrix('X'), T.vector('y'), updates='observe_updates')
    def _observe(self, X, y):
        pass

    def observe_updates(self, X, y):
        Kxx = T.nlinalg.matrix_inverse(self.kernel.distance(X, X))
        Kxxy = T.dot(Kxx, y)
        return [(self.Xtrain, X), (self.ytrain, y), (self.Kxx, Kxx),(self.Kxxy, Kxxy)]

    def sample(self, X, size=1):
        if self._observed:
            return self._sample_predictive(X, size)
        else:
            return self._sample_prior(X, size)

    @theanify(T.matrix('X'), T.iscalar('size'))
    def _sample_prior(self, X, size):
        return self.mvg(self._mean_prior(X), self._cov_prior(X), size)

    @theanify(T.matrix('X'), T.iscalar('size'))
    def _sample_predictive(self, X, size):
        return self.mvg(self._mean_predictive(X), self._cov_predictive(X), size)

if __name__ == "__main__":
    rbf = RBF(1.0)
    gp = GaussianProcess(rbf, 1)
    gp.compile()
