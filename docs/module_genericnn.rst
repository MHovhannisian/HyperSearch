:mod:`genericnn` — Generic interface to MLP modules
==========================================================

The unified interface to multilayer perceptron implementations in Python. Quickstart:

* Instantiate a class instance with an associated dataset ``X, Y``.
* Run :mod:`set_iter_settings` and :mod:`set_nn_settings` to specify non-default hyperparameters of the neural network.
* Run :mod:`run_test` to perform a fitting. Returns a dict with fitting history and performance measures.

Class reference
---------------

.. module:: genericnn
.. autoclass:: GenericNN
    :members:
