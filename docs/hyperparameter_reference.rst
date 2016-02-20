Hyperparameter reference
------------------------

The :mod:`unifiedmlp` module provides a uniform interface with which to build and test one-layer perceptrons. The :mod:`hypersearch` module relies upon this to sample model performance in hyperparameter-space.

.. role:: rubric
.. raw:: html

**Meta-hyperparameters** - *used before neural network is created*

.. _Keras: http://keras.io/
.. _Scikit-neuralnetworks: http://scikit-neuralnetwork.readthedocs.org/en/latest/index.html
.. _Scikit-learn: http://scikit-learn.org

+----------------+-----------------------+------------------------+-----------------------+
| Hyperparameter | Allowed values        | Translation            | Notes                 |
+================+=======================+========================+=======================+
|  module        | "keras"               | `Keras`_               | Python module to use. |
+                +-----------------------+------------------------+                       +
|                | "sknn"                |`Scikit-neuralnetworks`_|                       |
+                +-----------------------+------------------------+                       +
|                | "sklearn"             | `Scikit-learn`_        |                       |
+----------------+-----------------------+------------------------+-----------------------+
|  frac_training | *float*, (0,1]        | Portion of assigned training data to use.      |
+----------------+-----------------------+------------------------+-----------------------+



**Hidden layer hyperparameters**

.. _Rectified linear units: https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29
.. _Activation function: https://en.wikipedia.org/wiki/Activation_function

+----------------+----------------+-----------------------+----------------------------+------------------------+
| Hyperparameter | Allowed values | N L K                 | Translation                | Notes                  |
+================+================+=======================+============================+========================+
| hidden_units   | *int*, >0      |                       | Number of units in hidden layer                     |
+----------------+----------------+-----------------------+----------------------------+------------------------+
| activation     | relu           | ✓ ✓ ✓                 | `Rectified linear units`_  | `Activation function`_ |
+                +----------------+-----------------------+----------------------------+                        +
|                | linear         | ✓ ✗ ✓                 | Linear units               | in hidden layer.       |
+                +----------------+-----------------------+----------------------------+                        +
|                | logistic       | ✓ ✓ ✓                 | Logistic/Sigmoid units     |                        |
+                +----------------+-----------------------+----------------------------+                        +
|                | tanh           | ✓ ✓ ✓                 | Tanh units                 |                        |
+----------------+----------------+-----------------------+----------------------------+------------------------+
