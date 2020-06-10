"""
Class for Multivariate Linear and Logistic Regression.

Copyright (c) 2020 Gabriele Gilardi


X           (n_samples, n_inputs)           Input dataset (training)
Xp          (n_samples, n_inputs)           Input dataset (prediction)
Y           (n_samples, n_outputs)          Output dataset (training)
Yp          (n_samples, n_labels)           Output dataset (prediction)
W           (n_inputs, n_outputs)           Weight matrix
L2          scalar                          Regularization factor
J           scalar                          Cost function
grad        (n_var, )                       Unrolled gradient
theta       (n_var, )                       Unrolled weights

n_samples           Number of samples
n_inputs            Number of features in the original dataset plus 1
n_outputs           Number of labels/classes in the output dataset
n_labels            Number of outputs in the original dataset
n_var               Number of variables

Notes:
- input datasets <X> and <Xp> must include the column of 1s.
- the number of variables is (n_inputs*n_outputs).
- all gradients are returned unrolled.
"""

import numpy as np


def f_activation(z):
    """
    Numerically stable version of the sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z >= 0.0)
    a[idx] = 1.0 / (1.0 + np.exp(-z[idx]))

    idx = (z < 0.0)
    a[idx] = np.exp(z[idx]) / (1.0 + np.exp(z[idx]))

    return a


def logsig(z):
    """
    Numerically stable version of the log-sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z < -33.3)
    a[idx] = z[idx]

    idx = (z >= -33.3) & (z < -18.0)
    a[idx] = z[idx] - np.exp(z[idx])

    idx = (z >= -18.0) & (z < 37.0)
    a[idx] = - np.log1p(np.exp(-z[idx]))

    idx = (z >= 37.0)
    a[idx] = - np.exp(-z[idx])

    return a


def build_class_matrix(Y):
    """
    Builds the output array <Yout> for a classification problem. Array <Y> has
    dimensions (n_samples, 1) and <Yout> has dimension (n_samples, n_classes).
    Yout[i,j] = 1 specifies that the i-th sample belongs to the j-th class.
    """
    n_samples = Y.shape[0]

    # Classes and corresponding number
    Yu, idx = np.unique(Y, return_inverse=True)
    n_classes = len(Yu)

    # Build the array actually used for classification
    Yout = np.zeros((n_samples, n_classes))
    Yout[np.arange(n_samples), idx] = 1.0

    return Yout, Yu


class Regression:

    def __init__(self, problem=None, use_grad=True, L2=0.0):
        """
        problem         C = logistic regression, otherwise linear regression
        use_grad        True = calculate and return the gradient
        L2              Regolarizarion factor
        """
        self.problem = problem
        self.use_grad = use_grad
        self.L2 = L2
        self.W = np.array([])               # Weight matrix

        # For logistic regression only
        if (self.problem == 'C'):
            self.init_Y = True              # Flag to initialize Yout
            self.Yout = np.array([])        # Actual output
            self.Yu = np.array([])          # Class list

    def create_model(self, theta, args):
        """
        Creates the model for a linear/logistic regression problem.
        """
        # Unpack the arguments
        X = args[0]             # Input dataset
        Y = args[1]             # Output dataset

        # Build the weight matrix
        self.build_weight_matrix(theta, X.shape[1])

        # Logistic regression problem
        if (self.problem == 'C'):

            # The first time initialize Yout (output) and Yu (class list)
            if (self.init_Y):
                self.Yout, self.Yu = build_class_matrix(Y)
                self.init_Y = False

            # Cross-entropy cost function and gradient
            J, grad = self.cross_entropy_function(X)

        # Linear regression problem
        else:

            # Quadratic cost function and gradient
            J, grad = self.quadratic_function(X, Y)

        # If not used don't return the gradient
        if (self.use_grad):
            return J, grad
        else:
            return J

    def eval_data(self, Xp):
        """
        Evaluates the input dataset with the model created in <create_model>.
        """
        # Logistic regression problem
        if (self.problem == 'C'):

            # Activation values
            Z = Xp @ self.W
            A = f_activation(Z)

            # Most likely class
            idx = np.argmax(A, axis=1)
            Yp = self.Yu[idx].reshape((len(idx), 1))

        # Linear regression problem
        else:

            # Output value
            Yp = Xp @ self.W

        return Yp

    def build_weight_matrix(self, theta, n_inputs):
        """
        Builds the weight matrix from the array of variables.
        """
        n_outputs = len(theta) // n_inputs
        self.W = np.reshape(theta, (n_inputs, n_outputs))

    def cross_entropy_function(self, X):
        """
        Computes the cross-entropy cost function and the gradient (including
        the L2 regularization term).
        """
        n_samples = float(X.shape[0])
        tiny_number = 1.0e-10           # To avoid -inf in log

        # Activation
        Z = X @ self.W

        # Cost function  (activation value is calculated in the logsig function)
        error = (1.0 - self.Yout) * Z - logsig(Z)
        J = error.sum() / n_samples \
            + self.L2 * (self.W[1:, :] ** 2).sum() / (2.0 * n_samples)

        # Return if gradient is not requested
        if (self.use_grad is False):
            return J, 0.0

        # Gradient
        A = f_activation(Z)
        a0 = np.zeros((1, self.W.shape[1]))
        grad = X.T @ (A - self.Yout) / n_samples \
               + self.L2 * np.vstack((a0, self.W[1:, :])) / n_samples

        # Return the cost function and the unrolled gradient
        return J, grad.flatten()

    def quadratic_function(self, X, Y):
        """
        Computes the quadratic cost function and the gradient (including the
        L2 regularization term).
        """
        n_samples = float(X.shape[0])

        # Cost function
        error = X @ self.W - Y
        J = 1.0 / (2.0 * n_samples) \
            * ((error ** 2).sum() + self.L2 * (self.W[1:, :] ** 2).sum())

        # Return if gradient is not requested
        if (self.use_grad is False):
            return J, 0.0

        # Gradient
        a0 = np.zeros((1, self.W.shape[1]))
        grad = X.T @ error / n_samples \
               + self.L2 * np.vstack((a0, self.W[1:, :])) / n_samples

        # Return the cost function and the unrolled gradient
        return J, grad.flatten()
