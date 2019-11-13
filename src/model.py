import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from network_layer import GaussianLayer, HorseshoeLayer


def calculate_log_likelihood():
    """
    https://pdfs.semanticscholar.org/bd2a/fbef667369f52f2c57ec8e9f85470be85521.pdf
    https://link.springer.com/article/10.1007/s10994-016-5619-z
    """
    return 2


class GaussianBayes(tf.keras.Model):
    """
    https://github.com/thibo73800/tensorflow2.0-examples/blob/master/Create%20custom%20layer.ipynb

    

    """
    def __init__(self):
        super(GaussianBayes, self).__init__()

    def call(self, input):
        l1 = GaussianLayer(input)
        l2 = GaussianLayer(l1)
        return l2

    def elbo(self, phi):
        """p(D|thetha) = PI_n_n=1 p(y_n|x_n,thetha)
        L(phi) = E_phi (theta) [ log p(D|theta)] - KL [ q_phi(theta) || p(theta)]
        = E_phi (theta) [ log p(D)] - KL [ q(theta|phi) || p(theta|D)]
        """
        pass


class HorseshoeBayes(tf.keras.Model):
    """
    w|tau ~ N(0, tau^2)
    tau ~ C^+(0, b0)
    tau_j ~ C^+(0, b0)
    v ~ C^+(0, bg)
    Wij|tau_n,v ~ N(0, tau_j^2 * v^2)
    a ~ C^+(0, b)
    a|k ~ Inv Gamma(1/2, 1/k) 
    k ~ Inv Gamma(1/2, 1/b^2)
    b0 = 1
    """
    def __init__(self):
        super(GaussianBayes, self).__init__()

    def call(self, input):
        l1 = HorseshoeLayer(input)
        l2 = GaussianLayer(l1)
        return l2

    def prior(self, input):
       
        w_tau = tfp.distributions.Normal(input)
        return w_tau

    def tau():
        b0 = 1
        return tfp.distributions.HalfCauchy(0, b0)

