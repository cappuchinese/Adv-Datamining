"""Model programme for Chapter 2"""

__author__ = "Lisa Hu"
__date__ = 2022.03

from math import e


def linear(a: float) -> float:
    """
    Linear activation function
    :param a: x
    :return:
    """
    return a


def sign(a: float) -> float:
    """
    Signum activation function
    :param a: Input that needs classification
    :return: Corresponding class
    """
    if a < 0.0:
        return -1.0
    elif a > 0.0:
        return 1.0
    else:
        return 0.0


def tanh(a: float) -> float:
    """
    Hyperbolic tangent activation function
    :param a: x
    :return: tanh of a
    """
    e_a = e ** a
    e_nega = e ** -a
    return (e_a - e_nega) / (e_a + e_nega)


def mean_squared_error(yhat: float, y: float) -> float:
    """
    Mean squared error loss function
    :param yhat: Predicted y
    :param y: Target y
    :return: Squared loss
    """
    loss = (yhat - y) ** 2
    return loss


def mean_absolute_error(yhat: float, y: float) -> float:
    """
    Mean absolute error loss function
    :param yhat: Predicted y
    :param y: Target y
    :return: Absolute loss
    """
    loss = abs(yhat - y)
    return loss


def hinge(yhat: float, y: float) -> float:
    """
    Hinge loss function
    :param yhat:
    :param y:
    :return:
    """
    loss = max(1 - yhat * y, 0)
    return loss


def derivative(function, *, delta: float = 0.01):
    """
    Creates the derivative of input function
    :param function: Function to derive
    :param delta: Step length delta x
    :return: Derived function of input function
    """
    def wrapper_derivative(x, *args):
        dev = (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
        return dev

    wrapper_derivative.__name__ = function.__name__ + "`"
    wrapper_derivative.__qualname__ = function.__qualname__ + "`"
    return wrapper_derivative


class Neuron:
    """
    Attributes:
        dim (int): The number of dimensions
        activation (function): The activation function
        loss (function): The loss function
        bias (int): The bias value
        weights (list): The weights for each dimension accordingly
    """
    def __init__(self, dim: int, activation=linear, loss=mean_squared_error):
        """
        :param dim: Dimensions of the neuron
        :param activation: Activation function, default=linear
        :param loss: Loss function, default=mean squared error
        """
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        text = f"Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})"
        return text

    def predict(self, xs: list) -> list:
        """
        Prediction based on pre-activation function
        :param xs: List of instances
        :return: List of predictions
        """
        yhats = []
        # Go through the instances
        for x in xs:
            # Pre-activation
            pre_act = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            # Activation
            yhat = self.activation(pre_act)
            # Append yhat to list
            yhats.append(yhat)
        return yhats

    def partial_fit(self, xs: list, ys: list, *, alpha: float = 0.001):
        """
        Fitting the Neuron model
        :param xs: List of instances
        :param ys: List of targets
        :param alpha: Learning rate
        """
        # Go through the instances
        for x, y in zip(xs, ys):
            # Get the prediction
            pre_act = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            yhat = self.activation(pre_act)

            # Get derivatives
            activation_prime = derivative(self.activation)
            loss_prime = derivative(self.loss)
            u = loss_prime(yhat, y) * activation_prime(pre_act)

            # Update
            self.bias = self.bias - alpha * u
            self.weights = [wi - alpha * u * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, alpha: float = 0.001, epochs: int = 500):
        """
        Fitting until there are no more changes or max epochs are reached
        :param xs: List of instances
        :param ys: List of targets
        :param alpha: Learning rate
        :param epochs: Number of max runs
        """
        eps = 0
        updating = True
        while updating:
            # Get the old bias and weights
            old_bias = self.bias
            old_weights = self.weights

            # Update the bias and weights
            self.partial_fit(xs, ys, alpha=alpha)
            # Add epoch
            eps += 1
            # Check if the new bias and weights are different
            if old_bias == self.bias and old_weights == self.weights:
                print(f"Model is not updating anymore. Number of epochs run: {eps}")
                updating = False
            elif eps == epochs:
                print("Max number of epochs reached")
                updating = False
