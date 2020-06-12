# Multivariate Linear and Logistic Regression Using Gradient Descent Optimization

## Reference

- Mathematical background: [Linear Models in Scikit-Learn](https://scikit-learn.org/stable/modules/linear_model.html).

- Datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) regression implementation.
- Quadratic cost function for linear regression of continuous problems.
- Cross-entropy cost function for logistic regression of classification problems.
- Both cost functions include an L2-type regularization term.
- Classes in logistic regression are determined automatically.
- Sigmoid and cross-entropy function are computed using a numerically stable implementation.
- Option to not to compute and return the gradient of the cost function.
- Option to reduce the learning rate during the computation.
- A gradient descent optimizer (GDO) is included in *utils.py*, together with several other utility functions.
- The *Regression* class in *regression.py* is not constrained to the GDO solver but it can be used with any other optimizer.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (plant, stock, seed, wine.)

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

### Single-label linear regression example: plant

```python
data_file = 'plant_dataset.csv'
n_features = 4
split_factor = 0.7
L2 = 0.0
epochs = 5000
alpha = 0.1
d_alpha = 1.0
tolX = 1.e-10
tolF = None
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant>.

The dataset has 4 features, 1 label, 9568 samples, and 5 variables.

Closed-form solution: [4.54308846e+02, -1.48096084e+01, -2.89406829e+00,  4.11171081e-01, -2.29397325e+00].

Predicted/actual correlation values: 0.965 (training), 0.961 (test).

Exit on `tolX` after 2167 epochs.

### Multi-label linear regression example: stock

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

Original dataset: <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>.

The dataset has 7 features, 2 labels, 536 samples, and 16 variables.

Closed-form solution: [ 0.00073131,  0.00136731, -0.00056476, -0.00317299,  0.00135512,  0.00584854, -0.00018105, -0.00161590,  0.00545528,  0.00059722,  0.00671988,  0.00202262,  0.00014225,  0.00309809,  0.00059391,  0.00506524]

Predicted/actual correlation values: 0.944 (training), 0.945 (test).

Exit on `tolX` after 812 epochs.

### Multi-class logistic regression example: seed

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

Original dataset: <https://archive.ics.uci.edu/ml/datasets/seeds>.

The dataset has 7 features, 3 classes, 210 samples, and 24 variables.

Predicted/actual accuracy values: 97.3% (training), 95.2% (test).

Exit on `tolF` after 19387 epochs.

### Multi-class logistic regression example: wine

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

Original dataset: <https://archive.ics.uci.edu/ml/datasets/wine+quality>.

The dataset has 11 features, 6 classes, 1599 samples, and 72 variables.

Predicted/actual accuracy values: 60.7% (training), 57.9% (test).

Exit on `epochs` with `tolX` = 5.0e-5 and `tolF`= 2.0e-9.
