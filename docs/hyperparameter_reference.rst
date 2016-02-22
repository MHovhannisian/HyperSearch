Hyperparameter reference
------------------------

The :class:`~unifiedmlp.UnifiedMLP` class requires keyword arguments are passed to its :func:`~unifiedmlp.UnifiedMLP.set_hypers` method. The :class:`~hypersearch.HyperSearch` class requires the same arguments for its :func:`~hypersearch.HyperSearch.set_fixed` method, and the :func:`~hypersearch.HyperSearch.set_search` takes keywords with a range. This page contains the appropriate keywords and values that can be passed.

.. role:: rubric
.. raw:: html

**Using this reference**

* Default values are highlighted in bold or stated in the *value description*.
* The field *N L K* gives compatibiility in Scikit-\ **N**\ euralnetworks, Scikit-\ **L**\ earn, and **K**\ eras, respectively.

Meta-hyperparameters
""""""""""""""""""""

Used before neural network is created.

.. _Keras: http://keras.io/
.. _Scikit-neuralnetworks: http://scikit-neuralnetwork.readthedocs.org/en/latest/index.html
.. _Scikit-learn: http://scikit-learn.org

+----------------+---------------------------------+------------------------+------------------------+
| Hyperparameter |Description                      | Allowed values         | Value description      |
+================+=================================+========================+========================+
| module         |Python module to use.            | **"keras"**            | `Keras`_               |
+                +                                 +------------------------+------------------------+
|                |                                 | "sknn"                 |`Scikit-neuralnetworks`_|
+                +                                 +------------------------+------------------------+
|                |                                 | "sklearn"              | `Scikit-learn`_        |
+----------------+---------------------------------+------------------------+------------------------+
| frac_training  |Portion of training data to use. | *float*, (0,1]         | Default: 1.0           |
+----------------+---------------------------------+------------------------+------------------------+

|

Hidden layer hyperparameters
""""""""""""""""""""""""""""

.. _Rectified linear units: https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29
.. _Activation function: https://en.wikipedia.org/wiki/Activation_function

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| hidden_units   | Number of units in hidden layer | *int*, >0      | ✓ ✓ ✓ | Default: 15                |
+----------------+---------------------------------+----------------+-------+----------------------------+
| activation     | `Activation function`_          | **relu**       | ✓ ✓ ✓ | `Rectified linear units`_  |
+                + in hidden layer.                +----------------+-------+----------------------------+
|                |                                 | linear         | ✓ ✗ ✓ | Linear units               |
+                +                                 +----------------+-------+----------------------------+
|                |                                 | logistic       | ✓ ✓ ✓ | Logistic/Sigmoid units     |
+                +                                 +----------------+-------+----------------------------+
|                |                                 | tanh           | ✓ ✓ ✓ | Tanh units                 |
+----------------+---------------------------------+----------------+-------+----------------------------+

|

Regularisation
""""""""""""""

Mixing L2 and dropout is not supported in Scikit-Neuralnetworks.

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| alpha          | L2 penalty                      | *float*        | ✓ ✓ ✓ | Default: 0.0000            |
+----------------+---------------------------------+----------------+-------+----------------------------+
| dropout        | Dropout probability             | *float*, (0,1] | ✓ ✗ ✓ | Default: 0.0               |
+----------------+---------------------------------+----------------+-------+----------------------------+

|

Learning
""""""""

.. _Adadelta: http://sebastianruder.com/optimizing-gradient-descent/index.html#adadelta
.. _Adam: http://sebastianruder.com/optimizing-gradient-descent/index.html#adam

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| algorithm      | Learning algorithm              | **sgd**        | ✓ ✓ ✓ | Stochastic gradient descent|
+                +                                 +----------------+-------+----------------------------+
|                |                                 | adadelta       | ✓ ✗ ✓ | `Adadelta`_                |
+                +                                 +----------------+-------+----------------------------+
|                |                                 | adam           | ✗ ✓ ✓ | `Adam`_                    |
+----------------+---------------------------------+----------------+-------+----------------------------+
| learning_rate  | Learning rate (SGD and Adam)    | *float*, >0    | ✓ ✓ ✓ | Default: 0.001             |
+----------------+---------------------------------+----------------+-------+----------------------------+
| batch_size     | Samples per minibatch           | *int*, >0      | ✓ ✓ ✓ | Default: 16                |
+----------------+---------------------------------+----------------+-------+----------------------------+

|

SGD settings
""""""""""""

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| learning_decay | Per-epoch learning rate decay   | *float*        | ✓ ✓ ✓ | Default: 0.0               |
+----------------+---------------------------------+----------------+-------+----------------------------+
| momentum       | Momentum term                   | *float*, [0,1] | ✓ ✓ ✓ | Default: 0.9               |
+----------------+---------------------------------+----------------+-------+----------------------------+
| nesterov       | Nesterov's momentum             | *bool*         | ✓ ✓ ✓ | Default: False             |
+----------------+---------------------------------+----------------+-------+----------------------------+

|

Adam settings
"""""""""""""

The defaults are usually good options.

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| beta_1         | Decay rate, first moment vector | *float*, [0,1) | ✗ ✓ ✓ | Default: 0.0               |
+----------------+---------------------------------+----------------+-------+----------------------------+
| beta_2         | Decay rate second moment vector | *float*, [0,1) | ✗ ✓ ✓ | Default: 0.9               |
+----------------+---------------------------------+----------------+-------+----------------------------+
| epsilon        | Value for numerical stability   | *float*, [0,1) | ✗ ✓ ✓ | Default: False             |
+----------------+---------------------------------+----------------+-------+----------------------------+

|

Stopping criteria
"""""""""""""""""

Controlled uniformly by :mod:`unifiedmlp`, rather than the individual modules.

+----------------+---------------------------------+----------------+-------+----------------------------+
| Hyperparameter | Description                     | Allowed values | N L K | Value description          |
+================+=================================+================+=======+============================+
| max_epoch      | Maximum epochs before stopping  | *int*          | ✓ ✓ ✓ | Default: 100               |
+----------------+---------------------------------+----------------+-------+----------------------------+
| epoch_tol      | Tolerance on stopping criteria  | *float*        | ✓ ✓ ✓ | Default: 0.001             |
+----------------+---------------------------------+----------------+-------+----------------------------+
| n_stable       | Consecutive stable epochs       | *int*          | ✓ ✓ ✓ | Default: 3                 |
|                | before stopping                 |                |       |                            |
+----------------+---------------------------------+----------------+-------+----------------------------+
| early_stopping | Add performance on validation   | *bool*         | ✓ ✓ ✓ | Default: True              |
|                | data to stopping criteria.      |                |       |                            |
+----------------+---------------------------------+----------------+-------+----------------------------+
