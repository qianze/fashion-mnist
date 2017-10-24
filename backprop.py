import numpy as np


# Some helper functions.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# Helper function for intitializing weight matrices.
# you can put whatever weight intitialization scheme you want here.
def initialize_weights(d0, d1):
    return np.random.randn(d0, d1)


class NN(object):
    def __init__(self, layers):
        # a, the output with non-linearity applied.
        self.activations = []
        # z, the weighted output.
        self.z = []
        # W
        self.weights = []
        # The dimensions of our layers.
        self.layers = layers

        # Initialize the dimensions of our layers.
        for layer in layers:
            self.activations.append(np.zeros(layer))
            self.z.append(np.zeros(layer))

        # Initialize the weight values
        for i, layer in enumerate(layers[:-1]):
            self.weights.append(initialize_weights(layers[i + 1], layer))


    def feed_forward(self, inputs):
        # a^0 = x
        self.activation_input = np.array(inputs[:])

        # to keep track of generalized a^i
        a_m = self.activation_input
        self.activations[0] = a_m
        self.z[0] = a_m

        for i, next_weight in enumerate(self.weights):
            # z^(m + 1) = W^(m + 1)a^m
            z_m_next = next_weight.dot(a_m)
            # a^(m + 1) = f(z^(m + 1))
            a_m_next = sigmoid(z_m_next)

            self.activations[i + 1] = a_m_next
            self.z[i + 1] = z_m_next

            a_m = a_m_next


    def back_propagate(self, targets, learning_rate):
        # delta^L = f'(n^L) * (d J)/(d a)
        # (d J)/(d a) = a
        # In this case (d J)/(d a) = (a^L - t) as we are just using a basic
        # mean squared error.

        # Compute the error to get the final sensitivity term
        error = self.activations[-1] - targets
        delta_L= dsigmoid(self.activations[-1]) * error

        # Start with the final weight transformation
        m = len(self.weights) - 1

        while m >= 0:
            if m != len(self.layers) - 2:
                # delta^m = f'(z^m) * (W^(m+1))^T * detla^(m+1)
                f_prime = dsigmoid(self.activations[m + 1])
                delta_m = f_prime * self.weights[m + 1].T.dot(delta_m_next)
            else:
                delta_m = delta_L

            # W^m (k+1) = W^m(k) - \alpha delta^m * (a^(m-1))^T
            # Keep in mind k is fixed as this is a single iteration.
            self.weights[m] -= learning_rate * delta_m.dot(self.activations[m].T)
            delta_m_next = delta_m
            m -= 1

        error = 0.0

        # Compute the mean squared error.
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.activations[-1][k]) ** 2

        return error


    def train(self, patterns, iterations = 3000, learning_rate = 0.001):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]

                # Compute forward and backwards pass
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)

            print('%i: Error %.5f' % (i, error))

data = np.loadtxt('data/pima-indians-diabetes.data.txt', delimiter=',')
X, Y = data[:, :8], data[:, 8]

# Load the data into numpy arrays.
X = np.array(X)

# We have to pad with another dimension so we can mutliply by matrices.
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = np.array(Y)

# Same as above, we have to pad with another dimension to be able to multiply
# by matrices
Y = Y.reshape((Y.shape[0], 1))

# Hidden layer of 20 neurons
nn = NN([X.shape[1], 20, Y.shape[1]])
nn.train(list(zip(X, Y)))


