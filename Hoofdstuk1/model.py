"""Hoofdstuk 1 model"""

__author__ = "Lisa Hu"
__date__ = 2024.3


class Perceptron:
    def __init__(self, dim: int):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        return f"Perceptron(dim={self.dim})"

    def predict(self, xs: list) -> list:
        """
        Returns a list of predicted class label.
        :param xs: List of instances
        :return: List of predictions
        """
        yhats = []
        # Get individual instance
        for x in xs:
            # Pre-activation
            a = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            # Post-activation
            if a > 0.0:
                yhats.append(1.0)
            elif a < 0.0:
                yhats.append(-1.0)
            else:
                yhats.append(0.0)

        return yhats

    def partial_fit(self, xs: list, ys: list):
        """
        Fitting if there is an error
        :param xs: List of instances
        :param ys: List of labels
        """
        # Go through the instances
        for x, y in zip(xs, ys):
            yhat = self.predict([x])[0]

            # Update bias and weights when there is an error
            if error := yhat - y:
                self.bias = self.bias - error
                self.weights = [wi - error * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, epochs=0):
        """
        Executes fitting until the model is not updating anymore
        :param xs: List of instances
        :param ys: List of labels
        :param epochs: Number of maximum runs
        """
        eps = 0
        updating = True
        while updating:
            # Get the old bias and weights
            old_bias = self.bias
            old_weights = self.weights

            # Update the bias and weights
            self.partial_fit(xs, ys)
            # Add epoch
            eps += 1
            # Check if the new bias and weights are different
            if old_bias == self.bias and old_weights == self.weights:
                print(f"Model is not updating anymore. Number of epochs run: {eps}")
                updating = False
            # When max epochs is reached
            elif eps == epochs:
                print("Max number of epochs reached")
                updating = False


class LinearRegression:
    def __init__(self, dim: int):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        return f"Perceptron(dim={self.dim})"

    def predict(self, xs: list) -> list:
        yhats = []
        # Get individual instance
        for x in xs:
            # Pre-activation
            a = self.bias + sum(wi * xi for wi, xi in zip(self.weights, x))
            # Post-activation
            yhats.append(a)

        return yhats

    def partial_fit(self, xs: list, ys: list, *, alpha=0.01):
        # Go through the instances
        for x, y in zip(xs, ys):
            yhat = self.predict([x])[0]

            # Update bias and weights when there is an error
            if error := yhat - y:
                self.bias = self.bias - alpha * error
                self.weights = [wi - alpha * error * xi for wi, xi in zip(self.weights, x)]

    def fit(self, xs: list, ys: list, *, alpha: float = 0.01, epochs: int = 500):
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

