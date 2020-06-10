"""
Multivariate Linear and Logistic Regression

Copyright (c) 2020 Gabriele Gilardi


References
----------
- Mathematical background: linear models in Scikit-Learn
  @ https://scikit-learn.org/stable/modules/linear_model.html.

- Datasets: UCI Machine Learning Repository
  @ https://archive.ics.uci.edu/ml/datasets.php.

Characteristics
---------------
- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) regression implementation.
- Quadratic cost function for linear regression of continuous problems.
- Cross-entropy cost function for logistic regression of classification
  problems.
- Both cost functions include an L2-type regularization term.
- Classes in logistic regression are determined automatically.
- Sigmoid and cross-entropy function are computed using a numerically
  stable implementation.
- Option to not to compute and return the gradient of the cost function.
- Option to reduce the learning rate during the computation.
- A gradient descent optimizer (GDO) is included in <utils.py>, together
  with several other utility functions.
- The <Regression> class in <regression.py> is not constrained to the GDO
  solver but it can be used with any other optimizer.
- Usage: python test.py <example>.

Parameters
----------
example = house, stock, seed, wine
    Name of the example to run.
problem
    Defines the type of problem. Equal to C specifies logistic regression,
    anything else specifies linear regression. The default value is <None>.
use_grad = True, False
    Specifies if the gradient is calculated and returned. The default value
    is <True>.
data_file
    File name with the dataset (csv format).
n_features
    Number of features in the dataset (needed only for linear regression).
0 < split_factor < 1
    Split value between training and test data.
L2
    Regularization factor.
epochs
    Max. number of iterations (GDO.)
0 < alpha <= 1
    Learning rate (GDO.)
0 < d_alpha <= 1
    Rate decay of the learning rate (GDO.)
tolX, tolF
    Gradient absolute tolerance and function relative tolerance (GDO.) If both
    are specified the GDO will exit if either is satisfied. If both are not
    specified the GDO will exit when the max. number of iterations is reached.
"""

import sys
import numpy as np
import regression as reg
import utils as utl

# Read example to run
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

problem = None              # By default is a linear regression problem
use_grad = True             # By default calculate and return the gradient
np.random.seed(1294404794)

#  Single-label linear regression example
if (example == 'plant'):
    # https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
    # Dataset: 4 features, 1 label, 9568 samples, 5 variables
    # Closed-form solution:
    # [ 4.54308846e+02, -1.48096084e+01, -2.89406829e+00,  4.11171081e-01,
    #  -2.29397325e+00]
    # Correlation predicted/actual values: 0.965 (training), 0.961 (test).
    # Exit on tolX after 2167 epochs
    data_file = 'plant_dataset.csv'
    n_features = 4
    split_factor = 0.7
    L2 = 0.0
    epochs = 5000
    alpha = 0.1
    d_alpha = 1.0
    tolX = 1.e-10
    tolF = None

#  Multi-label linear regression example
elif (example == 'stock'):
    # https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
    # Dataset: 7 features, 2 labels, 536 samples, 16 variables
    # Closed-form solution:
    # [ 0.00073131,  0.00136731, -0.00056476, -0.00317299,  0.00135512,
    #   0.00584854, -0.00018105, -0.00161590,  0.00545528,  0.00059722,
    #   0.00671988,  0.00202262,  0.00014225,  0.00309809,  0.00059391,
    #   0.00506524]
    # Correlation predicted/actual values: 0.944 (training), 0.945 (test).
    # Exit on tolX after 812 epochs
    data_file = 'stock_dataset.csv'
    n_features = 7
    split_factor = 0.70
    L2 = 0.0
    epochs = 5000
    alpha = 0.2
    d_alpha = 1.0
    tolX = 1.e-7
    tolF = None

# Multi-class logistic regression example
elif (example == 'seed'):
    # https://archive.ics.uci.edu/ml/datasets/seeds
    # Dataset: 7 features, 3 classes, 210 samples, 24 variables
    # Accuracies predicted/actual values: 97.3% (training), 95.2% (test).
    # Exit on tolF after 19387 epochs
    data_file = 'seed_dataset.csv'
    problem = 'C'
    split_factor = 0.70
    L2 = 0.0
    epochs = 50000
    alpha = 0.9
    d_alpha = 1.0
    tolX = None
    tolF = 1.e-6

# Multi-class logistic regression example
elif (example == 'wine'):
    # https://archive.ics.uci.edu/ml/datasets/wine+quality
    # Dataset: 11 features, 6 classes, 1599 samples, 72 variables
    # Accuracies predicted/actual values: 60.7% (training), 57.9% (test).
    # Exit on epochs with tolX = 5.0e-5 and tolF= 2.0e-9
    data_file = 'wine_dataset.csv'
    problem = 'C'
    split_factor = 0.70
    L2 = 0.0
    epochs = 20000
    alpha = 0.99
    d_alpha = 1.0
    tolX = 1.e-15
    tolF = 1.e-15

else:
    print("Example not found")
    sys.exit(1)

# Read data from a csv file
data = np.loadtxt(data_file, delimiter=',')
n_samples, n_cols = data.shape

# Logistic regression (the label column is always the last one)
if (problem == 'C'):
    n_features = n_cols - 1
    n_labels = 1
    n_outputs, class_list = utl.get_classes(data[:, -1])

# Linear regression (the label columns are always at the end)
else:
    n_labels = n_cols - n_features
    n_outputs = n_labels

n_inputs = 1 + n_features               # Includes column of 1s
n_var = n_inputs * n_outputs

# Randomly build the training (tr) and test (te) datasets
rows_tr = int(split_factor * n_samples)
rows_te = n_samples - rows_tr
idx_tr = np.random.choice(np.arange(n_samples), size=rows_tr, replace=False)
idx_te = np.delete(np.arange(n_samples), idx_tr)
data_tr = data[idx_tr, :]
data_te = data[idx_te, :]

# Split the data
X_tr = data_tr[:, 0:n_features]
Y_tr = data_tr[:, n_features:]
X_te = data_te[:, 0:n_features]
Y_te = data_te[:, n_features:]

# Info
print("\nNumber of samples = ", n_samples)
print("Number of features = ", n_features)
print("Number of labels = ", n_labels)

print("\nNumber of inputs = ", n_inputs)
print("Number of outputs = ", n_outputs)
print("Number of variables = ", n_var)

if (problem == 'C'):
    print("\nClasses: ", class_list)

print("\nNumber of training samples = ", rows_tr)
print("Number of test samples= ", rows_te)

# Normalize training dataset and add column of 1s
Xn_tr, param = utl.normalize_data(X_tr)
X1n_tr = np.block([np.ones((rows_tr, 1)), Xn_tr])

# Initialize learner
learner = reg.Regression(problem=problem, use_grad=use_grad, L2=L2)

# Gradient descent optimization
func = learner.create_model
theta0 = np.zeros(n_var)
args = (X1n_tr, Y_tr)
theta, F, info = utl.GDO(func, theta0, args=args, epochs=epochs, alpha=alpha,
                         d_alpha=d_alpha, tolX=tolX, tolF=tolF)

# Results
print("\nCoeff. = ")
print(theta)
print("F = ", F)
print("Info = ", info)

# Evaluate training data
Yp_tr = learner.eval_data(X1n_tr)

# Normalize and evaluate test data
Xn_te = utl.normalize_data(X_te, param)
X1n_te = np.block([np.ones((rows_te, 1)), Xn_te])
Yp_te = learner.eval_data(X1n_te)

# Results for logistic regression (accuracy and correlation)
if (problem == 'C'):
    print("\nAccuracy training data = ", utl.calc_accu(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nAccuracy test data = ", utl.calc_accu(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))

# Results for linear regression (RMSE and correlation)
else:
    print("\nRMSE training data = ", utl.calc_rmse(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nRMSE test data = ", utl.calc_rmse(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))
    print("\nClosed-form solution:")
    print(utl.regression_sol(X1n_tr, Y_tr).flatten())
