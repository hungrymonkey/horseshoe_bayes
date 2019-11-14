import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from network_layer import GaussianLayer, HorseshoeLayer


def log_likelihood(input, mean, sigma):
    """
    //https://pdfs.semanticscholar.org/bd2a/fbef667369f52f2c57ec8e9f85470be85521.pdf
    https://link.springer.com/article/10.1007/s10994-016-5619-z
    https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/mle-regression.pdf
    https://www.statlect.com/glossary/log-likelihood

    likelihood = PI^N_n=1 p(y_n|x_n,theta)
    likelihood = PI^N_n=1 1/sqrt(2*pi*sigma^2) exp^(-(x-u)^2/(2*sigma^2))
    log(likelihood) = SUM^N_n=1 log(sqrt(2*pi*sigma^2)) + log(exp^(-(x-u)^2/(2*sigma^2)))
    log(likelihood) = -1/2 * log(2*pi*sigma^2) + -(x-u)^2/(2*sigma^2)

    log(f (xi; u, sig^2)) =
        -n/2 log(2*pi) - n/2 log(sig^2) - 1/(2sig)^2 sum(xi - u)^2

    log p(y_n|x_n, theta) =
        y_n log(f(x_n); theta) + (1-y_n)log(1 - f(x_n;theta))
    """
    n_samples = tf.shape(mean)[0]
    return (-n_samples * tf.math.log(sigma ** 2)
            - n_samples * tf.math.log(2 * np.pi) * .5
            - tf.reduce_sum((input - mean) ** 2) / (2 * sigma ** 2))


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
        """
        p(D|thetha) = PI_n_n=1 p(y_n|x_n,thetha)
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

    def elbo(self, phi):
        """
        L(phi)  = n/m sum^m_m=1 log p(y_m|x_m, theta) - KL[q_phi(theta) || p(theta)]

        theta = q_phi(theta), {(x_m, y_m)}^M_m=1 ~ D^m
        """
        pass

    def prior(self, input):
        """
        """
        w_tau = tfp.distributions.Normal(input)
        return w_tau

    def tau():
        b0 = 1
        return tfp.distributions.HalfCauchy(0, b0)

