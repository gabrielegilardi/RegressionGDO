# Multivariate Linear and Logistic Regression

## Reference

- Mathematical background: [linear model in Scikit-Learn](https://scikit-learn.org/stable/modules/linear_model.html).

- Datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) regression implementation.
- Quadratic cost function for linear regression of continuous problems.
- Cross-entropy cost function for logistic regression of classification problems.
- Both cost functions include an L2-type regularization term.
- Classes in logistic regression are determined automatically.
- Option to not to compute and return the gradient of the cost function.
- A gradient descent optimizer (GDO) is included in *utils.py*, together with several other utility functions.
- The *Regression* class in *regression.py* is not constrained to the GDO solver but it can be used with any other optimizer.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (house, stock, seed, wine.)

`problem` Defines the type of problem. Equal to C specifies logistic regression, anything else specifies linear regression. The default value is `None`.

`use_grad` Specifies if the gradient is calculated and returned. The default value is `True`.

`data_file` File name with the dataset (csv format).

`n_features` Number of features in the dataset (needed only for linear regression).

`split_factor` Split value between training and test data.

`L2` Regularization factor.

`epochs` Max. number of iterations (GDO).

`alpha` Learning rate (GDO).

`d_alpha` Rate decay of the learning rate (GDO).

`tolX`, `tolF` Gradient absolute tolerance and function relative tolerance (GDO). If both are specified the GDO will exit if either is satisfied. If both are not specified the GDO will exit when the max. number of iterations is reached.

## Examples

There are four examples in *test.py*: house, stock, seed, wine. Since GDO is used, `use_grad` is set to `True`.

### Single-label linear regression examples: house

```python
data_file = 'house_dataset.csv'
n_features = 2
split_factor = 0.7
L2 = 0.0
epochs = 5000
alpha = 0.99
d_alpha = 1.0
tolX = 1.e-7
tolF = None
```

The dataset has 2 features, 1 label, 47 samples, and 3 variables.

Closed-form solution: [335275.000, 113800.857, -3908.923].

Correlation predicted/actual values: 0.887 (training), 0.787 (test).

Exit on `tolX` after 52 epochs.

### Multi-label linear regression examples: stock

```python
data_file = 'stock_dataset.csv'
n_features = 7
split_factor = 0.70
L2 = 0.0
epochs = 5000
alpha = 0.2
d_alpha = 1.0
tolX = 1.e-7
tolF = None
```

The dataset has 7 features, 2 labels, 536 samples, and 16 variables.

Closed-form solution: [ 0.00073131,  0.00136731, -0.00056476, -0.00317299,  0.00135512,  0.00584854, -0.00018105, -0.00161590,  0.00545528,  0.00059722,  0.00671988,  0.00202262,  0.00014225,  0.00309809,  0.00059391,  0.00506524]

Correlation predicted/actual values: 0.944 (training), 0.945 (test).

Exit on `tolX` after 812 epochs.

### Multi-class logistic regression examples: seed

```python
data_file = 'seed_dataset.csv'
problem = 'C'
split_factor = 0.70
L2 = 0.0
epochs = 50000
alpha = 0.9
d_alpha = 1.0
tolX = None
tolF = 1.e-6
```

The dataset has 7 features, 3 classes, 210 samples, and 24 variables.

Accuracies predicted/actual values: 97.3% (training), 95.2% (test).

Exit on `tolF` after 19387 epochs.

### Multi-class logistic regression examples: wine

```python
data_file = 'wine_dataset.csv'
problem = 'C'
split_factor = 0.70
L2 = 0.0
epochs = 20000
alpha = 0.99
d_alpha = 1.0
tolX = 1.e-15
tolF = 1.e-15
```

The dataset has 11 features, 6 classes, 1599 samples, and 72 variables.

Accuracies predicted/actual values: 60.7% (training), 57.9% (test).

Exit on `epochs` with `tolX` = 5.0e-5 and `tolF`= 2.0e-9.
