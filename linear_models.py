import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegressionCustomized(object):
    """
    road_map:
        clf = LinearRegressionCustomized()
        clf.fit(X, y, test_size=0.2, random_state=5)
        ---> Validation score
        clf.predict(X)


    """
    def __init__(self, eta=0.4, epochs=500, w_choice='rand', test_size=0.33, random_state=5):
        self.eta = eta # learning rate
        self.epochs = epochs # number of iterations
        self.w_choice = w_choice # choice to initialize weights
        self.test_size = test_size # train/test ratio
        self.random_state = random_state

    def load_data(self, X, y):
        """load data into the model and split training/testing set"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.X_train = X_train.T
        self.y_train = np.array([y_train])
        self.X_val = X_val.T
        self.y_val = np.array([y_val])

    def param_init(self):
        """initialize weight and bias"""
        w = None
        if self.w_choice == 'rand':
            w = np.random.randn(1, self.len_w)
        elif self.w_choice == 'zero':
            w = np.zeros(1, self.len_w)
        b = 0
        return w, b

    def forward_prop(self, X, w, b):
        """forward propagation / predict target value"""
        # X: n x m
        # w: 1 x n
        z = np.dot(w, X) + b  # calculate dot product
        return z

    def cost_function(self, z, y):
        """define cost function that is going to be minimized"""
        m = y.shape[1]  # columns
        J = (1 / (2 * m)) * np.sum(np.square(z - y))
        return J

    def back_prop(self, X, y, z):
        """back propagation / finding the deltas for weight and bias"""
        m = y.shape[1]  # columns
        dz = (1 / m) * (z - y)
        dw = np.dot(dz, X.T)  # dim 1 x n
        db = np.sum(dz)
        return dw, db

    def gradient_descent_update(self, w, b, dw, db, eta):
        """update weight and bias through gradient descent"""
        w = w - eta * dw
        b = b - eta * db
        return w, b

    def fit(self, X, y):
        """(most important method) train and validate"""
        self.load_data(X, y) # load and split dataset
        self.len_w = self.X_train.shape[0] # def length of weight vector
        self.w, self.b = self.param_init() # initialize weight and bias

        self.costs_train = [] # to store training cost
        m_train = self.y_train.shape[1]
        m_val = self.y_val.shape[1]

        z_val = None
        for i in range(1, self.epochs + 1):
            z_train = self.forward_prop(self.X_train, self.w, self.b)
            self.cost_train = self.cost_function(z_train, self.y_train)
            dw, db = self.back_prop(self.X_train, self.y_train, z_train)
            self.w, self.b = self.gradient_descent_update(self.w, self.b, dw, db, self.eta)

            # store training cost in a list for plotting purposes
            if i % 10 == 0:
                self.costs_train.append(self.cost_train)

            # MAE_train
            self.MAE_train = (1 / m_train) * np.sum(np.abs(z_train - self.y_train))

            #calculate cost_val mean absolute error for validation set
            z_val = self.forward_prop(self.X_val, self.w, self.b)
            self.cost_val = self.cost_function(z_val, self.y_val)
            self.MAE_val = (1 / m_val) * np.sum(np.abs(z_val - self.y_val))

            # print report
            print('Epochs ' + str(i) + '/' + str(self.epochs) + ': ')
            print('Training cost: ' + str(self.cost_train) + '|' + 'Validation Cost: ' + str(self.cost_val))
            print('Training MAE: ' + str(self.MAE_train) + '|' + 'Validation MAE: ' + str(self.MAE_val))

        # plot
        self.plot_cost()

        # r^2
        print(self.y_val)
        self.r_2 = 1 - sum((z_val.flatten() - self.y_val.flatten()) ** 2) / sum((self.y_val.flatten() - np.mean(self.y_val.flatten())) ** 2)
        #self.r_2 = r2_score(self.y_val, z_val)


        return self

    def plot_cost(self):
        """Plot cost over iterations"""

        plt.plot(self.costs_train)
        plt.xlabel('Iterations per tens')
        plt.ylabel('Training Cost')
        plt.title('Learning rate ' + str(self.eta))
        plt.show()

        self.filepath = 'result/eta%10.2f_' % self.eta
        plt.savefig((self.filepath + 'cost_plot.png').replace(' ', ''))

    def predict(self, X_val):
        """return the predicted price using forward propagation"""
        self.len_w = X_val.shape[0]
        X_val = X_val.T
        return self.forward_prop(X_val, self.w, self.b)

    def get_R2_score(self):
        """Calculate R^2 as a way to evaluate the model"""
        return self.r_2

