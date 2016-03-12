Installation
------------

.. _virtualenv: https://pypi.python.org/pypi/virtualenv

The nature of HyperSearch means that it has a number of dependencies. You may like to set up a `virtualenv`_ for this module.

Prerequisites
"""""""""""""

.. _scikit-neuralnetwork: http://scikit-neuralnetwork.readthedocs.org/en/latest/guide_installation.html
.. _scikit-learn: http://scikit-learn.org/dev/developers/advanced_installation.html#install-bleeding-edge
.. _keras: http://keras.io/#installation

The module has been tested against library versions which are the newest stable versions or newer, specifically:

* Numpy 1.10.4:
     >>> pip install numpy
* Scipy 0.17.0:
     >>> pip install scipy
* Theano 0.8.0.dev0:
     >>> pip install git+git://github.com/Theano/Theano.git
* Cython 0.23.4:
     >>> pip install cython
* Progressbar 2.3:
     >>> pip install progressbar
* Matplotlib 1.3.1:
     >>> pip install matplotlib
* Seaborn 0.7.0:
     >>> pip install seaborn

**Neural network modules**

* Scikit-learn 0.18.dev0:
     See `scikit-learn`_ documenation.
* Pylearn 0.1.dev0:
     >>> pip install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package
* Keras 0.3.1:
     See `keras`_ documentation.
* Scikit-neuralnetwork 0.6.1:
     See `scikit-neuralnetwork`_ documentation; follow the Pulling Repositories method.

Installation
""""""""""""

Install the newest version of the code:

>>> pip install git+git://github.com/SCLElections/HyperSearch.git
