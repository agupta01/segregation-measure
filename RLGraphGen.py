import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Add
"""
UPDATE: Use PyTorch Geometric!
https://pytorch-geometric.readthedocs.io/en/latest/index.html
NOTE FOR FUTURE: Consider adding a random*weight layer to each model.
Possible that the model will train that layer's weight down to 0 eventually,
but it will act as a constantly decreasing noise factor to the model that is
proportional to the model's progress in training. This could potentially help
with stuck local minima (as the cost of being stuck in a minima is higher the
further a model is from convergence).
"""

def evalulate(self):
    """
    Evaluates output of graph to compute loss based on penalty-reward
    loss function (defined arbitrarily).
    """

class RandLayer(keras.layers.Layer):
    def __init__(self):
        super(RandLayer, self).__init__()

    def call(self, inputs):
        randWeights = np.array([np.random.rand(inputs.shape[0])]).T
        return tf.convert_to_tensor(inputs.numpy() * randWeights)

class RLGraphNet(keras.Model):
    def __init__(self):
        super(RLGraphNet, self).__init__()
        self.dense1 = Dense()
        self.dense2 = Dense()
        self.rand1 = RandLayer()
        self.dense3 = Add(self.dense2, self.rand1)
        self.dense4 = Dense()
        self.generator = Dense()

    def call(self, inputs):
        x = self.dense1(inputs)
        x_1 = self.dense2(x)
        x_2 = self.rand1(x)
        x = self.dense3(x_1, x_2)
        x = self.dense4(x)
        return self.generator(x)

def main():
    inputs = tf.ones((5,1))
    linear_layer = RandLayer()
    print(linear_layer(inputs))

if __name__ == '__main__':
    main()
