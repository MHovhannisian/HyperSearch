.. toctree::

:mod:`unifiedmlp` — Generic interface to MLP modules
====================================================

The unified interface to multilayer perceptron implementations in Python.

Quick-start:

* Instantiate a :class:`~unifiedmlp.UnifiedMLP` instance with an associated dataset ``X, Y``.
* Run :func:`~unifiedmlp.UnifiedMLP.set_hypers` to specify non-default hyperparameters of the neural network.
* Run :func:`~unifiedmlp.UnifiedMLP.run_test` to perform a fitting. Returns a dict with fitting history and performance measures.

Please also check the :doc:`hyperparameters reference guide<../hyperparameter_reference>`.

Class reference
---------------

.. module:: unifiedmlp
.. autoclass:: UnifiedMLP
    :members:

.. _results-dict:

Results dict reference
----------------------

The keys of the dict are shown below with brief explanations. Nested dicts represented as nested bullet points. Quantities end in _all when they are not per-class.

* **hypers**: Complete dict of the hyperparameters under which the model was built, trained and tested.

* **training**: Lists of model properties at end of each epoch.

  * **accuracy**: *List of lists.* Per-class accuracy on validation dataset.
  * **F1**: *List of lists.* Per-class F1 score on validation dataset.
  * **loss_all**: *List.* Training loss on the training dataset.
  * **time_all**: *List.* Wallclock training time in seconds.
  * **accuracy_all**: *List.* Accuracy on validation dataset.
  * **F1_all**: *List.* F1 score on validation dataset.

* **performance**: Model properties after training.

  * **accuracy**: *List.* Per-class accuracy on test dataset.
  * **F1**: *List.* Per-class F1 score on test dataset.
  * **time_all**: *Float.* Average training time in seconds for one epoch.
  * **accuracy_all**: *Float.* Accuracy on test dataset.
  * **F1_all**: *Float.* F1 score on validation dataset.
  * **n_epochs_all**: *Int.* Epochs taken to reach training convergence.
