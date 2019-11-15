import numpy as np


class LinearRegression(object):

    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta  # learning rate
        self.n_iter = n_iter  # iterations

    def fit(self, X, y):
        """Model training"""
        X = np.insert(X, 0, 1, axis=1)  # insert x0 into the first column
        self.w = np.zeros(X.shape[1])  # initialize weight to 1
        m = X.shape[0]  # X is a 'm x n' matrix

        # Gradient Descent
        for _ in range(self.n_iter):
            output = X.dot(self.w)  # calculate dot product of input features and weight
            errors = (y - output) **2  # difference between target value and predicted value
            self.w += self.eta / (2*m) * errors.dot(X)  # update weight based on gradient descent

        return self

    def predict(self, X):
        """Model testing"""
        return np.insert(X, 0, 1, axis=1).dot(self.w)  # insert x0 and update weight based on gradient for testing set

    def score(self, X, y):
        """Calculate R^2 as a way to evaluate the model"""
        r_2 = 1 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)
        return r_2


class LinearRegressionSGD(object):

    def __init__(self, eta=0.1, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # x1 in the first column
        self.w = np.ones(X.shape[1])  # initialize weight to 1

        # Gradient Descent
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for x, target in zip(X, y):
                output = x.dot(self.w)
                error = target - output
                self.w += self.eta * error * x

        return self

    def _shuffle(self, X, y):
        r = np.random.permutation((len(y)))
        return X[r], y[r]

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)


class LinearRegressionNormal(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)
