.. toctree::

:mod:`unifiedmlp` — Generic interface to MLP modules
====================================================

The unified interface to multilayer perceptron implementations in Python.

Quickstart:

* Instantiate a class instance with an associated dataset ``X, Y``.
* Run :mod:`set_iter_hypers` and :mod:`set_nn_hypers` to specify non-default hyperparameters of the neural network.
* Run :mod:`run_test` to perform a fitting. Returns a dict with fitting history and performance measures.

Please also check the :doc:`hyperparameters reference guide<../settings_reference>`.

Class reference
---------------

.. module:: unifiedmlp
.. autoclass:: UnifiedMLP
    :members:

.. _results-dict:

Results dict reference
----------------------

Hello
