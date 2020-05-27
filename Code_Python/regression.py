"""
Class for Multivariate Linear and Logistic Regression.

Copyright (c) 2020 Gabriele Gilardi


Common quantities:
    X           (n_samples, 1+n_features)           Input dataset (training)
    Xp          (n_samples, 1+n_features)           Input dataset (prediction)
    L2          scalar                              Regularization factor
    J           scalar                              Cost function

For logistic regression:
    theta       (1+n_features * classes, )          Unrolled weight matrix
    W           (1+n_features, n_classes)           Weight matrix
    Y           (n_samples, )                       Output classes
    Yp          (n_samples, )                       Predicted output classes
    grad        (1+n_features, n_classes)           Gradient

For linear regression:
    theta       (1+n_features * n_labels, )         Unrolled weight matrix
    W           (1+n_features, n_labels)            Weight matrix
    Y           (n_samples, n_labels)               Output labels
    Yp          (n_samples, n_labels)               Predicted output labels
    grad        (1+n_features, n_labels)            Gradient

Notes:
- input datasets <X> and <Xp> must include the column of 1s.
- all gradients are returned unrolled.
"""

import numpy as np


def f_activation(z):
    """
    Activation function (sigmoid)
    """
    a = 1.0 / (1.0 + np.exp(-z))
    return a


def build_class_matrix(Y):
    """
    Build the output array <Yout> for a classification problem. Array <Y> has
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
        Create the model for a linear/logistic regression problem.
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

            # Cross-entropy cost function and the gradient
            J, grad = self.cross_entropy_function(X)

        # Linear regression problem
        else:

            # Quadratic cost function and the gradient
            J, grad = self.quadratic_function(X, Y)

        # If not used don't return the gradient
        if (self.use_grad):
            return J, grad
        else:
            return J

    def eval_data(self, Xp):
        """
        Evaluate the input dataset with the model created in <create_model>.
        """
        # Logistic regression problem
        if (self.problem == 'C'):

            # Activation values
            Z = Xp @ self.W
            A = f_activation(Z)

            # Most likely class
            idx = np.argmax(A, axis=1)
            Yp = self.Yu[idx]

        # Linear regression problem
        else:

            # Output value
            Yp = Xp @ self.W

        return Yp

    def build_weight_matrix(self, theta, n_inputs):
        """
        Build the weight matrix from the array of variables.
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
        A = f_activation(Z)
        A = np.fmin(np.fmax(A, tiny_number), (1.0 - tiny_number))

        # Cost function
        error = self.Yout * np.log(A) + (1.0 - self.Yout) * np.log(1.0 - A)
        J = - error.sum() / n_samples \
            + self.L2 * (self.W[1:, :] ** 2).sum() / (2.0 * n_samples)

        # Return if gradient is not requested
        if (self.use_grad is False):
            return J, 0.0

        # Gradient
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
