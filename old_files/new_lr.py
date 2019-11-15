import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)
    # w = np.zeros(1, lenw)
    b = 0
    return w, b


def forward_prop(X, w, b):
    # X: n x m
    # w: 1 x n
    z = np.dot(w, X) + b  # 1 x m // b vector (b, b, b, b, b)
    return z


def cost_function(z, y):
    m = y.shape[1]  # columns
    J = (1 / (2 * m)) * np.sum(np.square(z - y))
    return J


def back_prop(X, y, z):
    m = y.shape[1]  # columns
    dz = (1 / m) * (z - y)
    dw = np.dot(dz, X.T)  # dim 1 x n
    db = np.sum(dz)
    return dw, db


def gradient_descent_update(w, b, dw, db, eta):
    # 8:49
    w = w - eta * dw
    b = b - eta * db
    return w, b


def linear_regression_model(X_train, y_train, X_val, y_val, eta, epochs):
    lenw = X_train.shape[0]
    w, b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1, epochs + 1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(X_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, eta)

        # store training cost in a list for plotting purposes
        if i % 10 == 0:
            costs_train.append(cost_train)

        # MAE_train
        MAE_train = (1 / m_train) * np.sum(np.abs(z_train - y_train))

        # cost_val MAE_val
        z_val = forward_prop(X_val, w, b)
        cost_val = cost_function(z_val, y_val)
        MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

        # print report
        print('Epochs ' + str(i) + '/' + str(epochs) + ': ')
        print('Training cost ' + str(cost_train) + '|' + 'Validation Cost ' + str(cost_val))
        print('Training MAE ' + str(MAE_train) + '|' + 'Validation MAE ' + str(MAE_val))

    # plot
    plt.plot(costs_train)
    plt.xlabel('Iterations per tens')
    plt.ylabel('Training Cost')
    plt.title('Learning rate ' + str(eta))
    plt.show()