# import numpy as np
import tensorflow as tf
from tensorflow_probability.keras import layers


class GaussianLayer(layers.Layers):
    """
    Normal Gaussian Baynesian Layer
    p(D|th) PI N_n=1 p(y_n| x_n, th)

    """
    def __init__(self, in_features, ouput_features, arguments, name=None):
        super(GaussianLayer, self).__init__(name=name)

    def call(self, inputs):
        return layers.Dense(inputs)
        pass


class HorseshoeLayer(tf.Module):
    """
    https://www.tensorflow.org/api_docs/python/tf/Module
    """
    def __init__(self, in_features, ouput_features, arguments, name=None):
        super(HorseshoeLayer, self).__init__(name=name)

    def call(self, parameter_list):
        pass
