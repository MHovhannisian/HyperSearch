#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core Python modules
import pickle
import itertools
import warnings
import collections
import sys
import csv

# User-contributed modules
import numpy as np
import progressbar

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import unifiedmlp


class HyperSearch(object):
    ''' Perform hyperparameter grid-search over multiple MLP implementations.

    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        Vectors of features for each sample, where there are n_samples vectors
        each with n_features elements.

    Y : array-like, shape (n_samples, n_classes)
        Vectors of labelled outcomes for each sample. UnifiedMLP expects a
        boolean or binary array specifying membership to each of n_classes
        classes.

    class_names : list, shape (n_classes)
        Ordered list of names by which to refer to each outcome class for
        per-class outputting of neural network performance.

    split : tuple of 3 entries, summing to 1.0 or less.
        The split of data between training, validation and testing. Training
        data is passed to fit() methods, validation data is used to track
        fitting progress and can be used for early stopping, and test data is
        used for the final evaluation of model quality.

    Examples
    --------

    >>> hs = HyperSearch(X, Ys, class_names=class_names)
    >>> hs.set_fixed(algorithm='adam')
    >>> hs.set_search(
    ...     module=['keras', 'sklearn'],
    ...     dropout=np.arange(0.0, 1.0, 0.2),
    ...     batch_size=range(12, 100, 24),
    ...     hidden_units=[10, 15]
    ... )
    Registered 80 tests to run.
    >>> hs.run_tests(ignore_failure=True).save()
    Some tests failed due to incompatible settings:
    Module  | Count | Error message
    sklearn |    32 | Unsupported settings: dropout
    [...]

    >>> hs = HyperSearch.load().summary()
    [...]
    >>> hs.graph(y_axis="accuracy", x_axis="dropout", batch_size='*')
    hypersearch.py:331: UserWarning: Unspecified parameters were automatically fixed:
        hidden_units: 10
        module: keras

    '''

    def __init__(self, X, Y, class_names=[], split=(0.70, 0.15, 0.15)):

        X = np.array(X).astype('float64')
        Y = np.array(Y).astype(bool)

        self.n_classes = Y.shape[1]

        if class_names:
            self.cls = tuple(class_names)
        else:
            self.cls = tuple([str(i) for i in range(self.n_classes)])

        self.MLP = unifiedmlp.UnifiedMLP(X, Y, split)

        self._dim_names = []
        self._dim_vals = []

        self.fixed = False
        self.search = False

        # Data on outputting
        self.results_ylabels = {'F1': 'F1 score (test data)',
                                'accuracy': 'Accuracy score (test data)',
                                'time': 'Training time per epoch (seconds)',
                                'n_epochs': 'Training epochs before stopping'
                                }
        self.perclass_ylabels = ['F1', 'accuracy']
        self.training_ylabels = {'F1': 'F1 score (validation data)',
                                 'accuracy': 'Accuracy score (validation data)',
                                 'time': 'Training time for epoch (seconds)',
                                 'loss': 'Loss function (training data)'
                                 }

        self.xlabels = {
            'module': 'Python module',
            'frac_training': 'Fraction training data used',

            'hidden_units': 'Number of hidden units',
            'activation': 'Activation function in hidden layer',

            'alpha': 'L2 penalty (alpha)',
            'dropout': 'Dropout (probability)',

            'algorithm': 'Learning algorithm',
            'learning_rate': 'Learning rate',
            'batch_size': 'Samples per minibatch',

            'learning_decay': 'Per-epoch learning rate decay (SGD)',
            'momentum': 'Momentum term (SGD)',
            'nesterovs_momentum': 'Nesterov\'s momentum (SGD)',

            'beta_1': 'beta_1 parameter (Adam)',
            'beta_2': 'beta_2 parameter (Adam)',
            'epsilon': 'epsilon parameter (Adam)',

            'max_epoch': 'Maximum epochs before stopping',
            'epoch_tol': 'Tolerance on stopping criteria',
            'n_stable': 'Consecutive stable epochs before stopping',
            'early_stopping': 'Performance on validation data as stopping criterion',

            'epoch' : 'Epoch'
        }

        self.errs = collections.Counter()

    def set_fixed(self, **hypers):
        ''' Set hyperparameters to keep constant through the parameter sweep.

        Returns
        -------

        self

        Examples
        --------

        >>> hs.set_fixed(module='sklearn')

        >>> hs.set_fixed(module='keras', hidden_units=20)

        >>> fixed_settings_dict = {'module': 'keras', 'algorithm': 'adam'}
        >>> hs.set_fixed(**fixed_settings_dict)
        '''

        self.fixed = True
        self.MLP.set_hypers(**hypers)

        return self

    def set_search(self, **hypers):
        ''' Set hyperparameter ranges to search over.

        Can only be run once per instance.

        Returns
        -------

        self

        Examples
        --------

        >>> import numpy as np
        >>> from hypersearch import HyperSearch
        >>> hs = HyperSearch(X, Y)
        >>> hs.set_search(
        ...     module=['sknn', 'keras', 'sklearn'],
        ...     dropout=np.arange(0.1, 1.0, 0.1),
        ...     algorithm=['adam', 'adadelta']
        ... )
        Registered 60 tests to run.

        '''
        assert(not self.search)
        self.search = True

        for key, range_ in hypers.items():
            self._dim_names.append(key)
            self._dim_vals.append(np.sort(np.array(range_)))

        self._n_dims = len(self._dim_names)
        self._dims = np.array([len(self._dim_vals[i])
                               for i in range(self._n_dims)])
        self._dim_idxs = tuple(
            range(len(self._dim_vals[i])) for i in range(self._n_dims))

        self.results = np.empty(self._dims, dtype=object)
        self.ran = np.zeros(self._dims, dtype=bool)
        self.success = np.ones(self._dims, dtype=bool)

        self.accuracy_all = np.zeros(self._dims, dtype=float)
        self.F1_all = np.zeros(self._dims, dtype=float)

        # Keep separate results for each class
        # Add an extra inner dim to the arrays
        self.accuracy = np.zeros(tuple(self._dims) + tuple([self.n_classes]),
                                 dtype=float)
        self.F1 = np.zeros(tuple(self._dims) + tuple([self.n_classes]),
                           dtype=float)

        print "Registered", np.prod(self._dims), "tests to run."

        return self

    def run_tests(self, ignore_failure=True):
        ''' Execute the specified parameter search.

        Parameters
        ----------

        ignore_failure : bool
            Continue past tests which fail due to incompatible settings. If
            True, missing data will be automatically excluded from graphs
            and made clear in csv outputs. Errors raised will be printed for
            review at the end.

        Returns
        -------

        self

        '''

        try:
            assert (self.search)
        except AssertionError:
            raise AssertionError('No parameters to search specified.')

        self.static_hyperparams = self._get_static_hyperparams(
            self.MLP.get_hypers(), self._dim_names)

        errs = collections.Counter()
        n_tests = np.prod(self._dims)

        bar = progressbar.ProgressBar(
            maxval=n_tests,
            widgets=[
                progressbar.ETA(),
                progressbar.Bar(),
                progressbar.Counter(format='%d of ' + "{}".format(n_tests))
            ]).start()

        for i, coordinates in enumerate(itertools.product(*self._dim_idxs)):
            if self.ran[coordinates]:
                continue

            this_settings = {self._dim_names[j]: self._dim_vals[j][coord]
                             for j, coord in enumerate(coordinates)}
            self.MLP.set_hypers(**this_settings)
            module = self.MLP.get_hypers()['module']

            try:
                result, hypers, _ = self.MLP.run_test()
                self.success[coordinates] = True

                self.results[coordinates] = result

                self.accuracy_all[coordinates] = result[
                    'performance']['accuracy_all']
                self.F1_all[coordinates] = result['performance']['F1_all']

                self.accuracy[coordinates] = result['performance']['accuracy']
                self.F1[coordinates] = result['performance']['F1']

            except KeyError as e:  # When unsupported settings are specified
                if ignore_failure:
                    self.success[coordinates] = False
                    err = (module, e.message)
                    errs[err] += 1
                else:
                    raise
            finally:
                self.ran[coordinates] = True

            bar.update(i + 1)

        bar.finish()

        self._show_errs(errs)
        self.errs.update(errs)

        # Keep one record of the hyperparamters. We know which ones were
        # changed between runs.
        self.hypers = hypers

        return self

    def save(self, file_name="auto.save"):
        ''' Save the state of the HyperSearch instance. '''

        with open(file_name, 'w') as pickle_file:
            pickle.dump(self, pickle_file)

        return self

    @staticmethod
    def load(file_name="auto.save"):
        ''' Load a previously saved HyperSearch instance. '''

        with open(file_name, 'r') as pickle_file:
            instance = pickle.load(pickle_file)

        return instance

    def graph(self, x_axis='epoch', y_axis='accuracy', z_axis='accuracy',
              x_log=False, y_log=False, classes=None, **vals):
        ''' Visualise a slice of the data through a graph.

        If the slice to take is underspecified, unconstrained hyperparameters
        are set to their value in the neural network with maximum accuracy.

        Parameters
        ----------

        x_axis : str, 'epoch' or any hyperparameter from the search.
            If 'epoch', show how MLP performance changed during training.
            If a hyperparameter, show MLP performance between models varying
            by the given hyperparameter.

        y_axis : str.
            If **x_axis** = 'epoch', y_axis is an array of per-epoch data,
            one of:

                * *accuracy* or *F1* - performance on the validation set;
                * *loss* - loss function on the training data;
                * *time* - epoch's wallclock training time (seconds).

            If **x_axis** = hyperparameter, y_axis is a scalar, one of:

                * *accuracy* or *F1* - performance on the test set;
                * *time* - average training time per epoch
                * *n_epochs* - number of epochs taken to reach convergence.

        z_axis : str, {accuracy, F1, time, n_epochs}
            If **y_axis** and **x_axis** are both hyperparameters, a performance
            metric can be specified on the z_axis to create a 3D plot. In this
            case, options to plot multiple lines are not supported.

        x_log : bool
            Use logarithmic scale on the x-axis.

        y_log : bool
            For 3D graphs only. Use logarithmic scale on the y-axis.

        classes : None or str or list of str or '*'
            If not None, plot separate lines for each given class, according to
            class names given at initialisation. Applies to **y_axis** = F1 or
            accuracy.

            '*' specifies all classes.

        vals : Key-value pairs. Values can also be a list or '*'.
            Constrain the slice of the dataset to take by fixing hyperparameters
            e.g. `dropout=0.4`. Provide a list or '*' to prouce a seperate line
            for each of multiple or all hyperparameter values.

        Examples
        --------

        Results graph (x_axis set to a hyperparameter)

        >>> hs.graph(y_axis="n_epochs", x_axis="batch_size")
        UserWarning: Unspecified parameters were automatically fixed:
            dropout: 0.4
        >>> hs.graph(y_axis="accuracy", x_axis="dropout", batch_size='*')

        Training graph (x_axis set to "epoch")

        >>> hs.graph(y_axis="time", x_axis="epoch", module='sklearn', dropout=0.0, hidden_units='*')
        >>> hs.graph(y_axis="F1", x_axis="epoch", classes=['brown_hair', 'over35', 'city_dweller'])

        3D results graph (x_axis and y_axis set to hyperparameters)

        >>> hs.graph(y_axis="dropout", x_axis="batch_size")

        '''

        # Setup

        if not self.ran.all():
            print "Cannot make graphs - please run run_tests()."
            return self

        # Homogenise inputs into a list of raw vals.
        # Careful typing here to avoid ambiguities
        for key, item in vals.items():
            try:
                assert(key != x_axis)
            except AssertionError:
                err = "A keyword argument is the same as the x_axis dimension."
                raise AssertionError(err)

            if item == '*':
                dim = self._dim_names.index(key)
                vals[key] = list(self._dim_vals[dim])
            elif (isinstance(item, float) or isinstance(item, int) or
                    isinstance(item, basestring)):
                vals[key] = [item]
            elif hasattr(item, "__getitem__"):
                pass
            else:
                errstr = ("Pass hyperparams as either a raw value, a list of" +
                          " raw values, or '*'.")
                raise TypeError(errstr)

        if classes == '*':
            classes = list(self.cls)
        elif isinstance(classes, basestring):
            classes = [classes]
        elif classes:
            classes = [str(c) for c in classes]
        else:
            classes = []

        # Call appropriate graphing function
        if x_axis == 'epoch':
            self._training_graph(y_axis, x_log, classes, **vals)
        elif y_axis not in self.results_ylabels.keys():
            try:
                assert(z_axis in self.results_ylabels.keys())
            except AssertionError:
                raise AssertionError("A performance metric must be on the" +
                                     " y-axis for 2D graphs or z-axis for 3D graphs.")
            self._results_graph3D(x_axis, y_axis, x_log, y_log, z_axis, **vals)
        else:
            self._results_graph(x_axis, y_axis, x_log, classes, **vals)

        return self

    def _training_graph(self, y_axis, x_log, classes=[], **vals):
        ''' Graph test/validation performance *per epoch*.

        Where the parameters `lines`, and `**vals` not been used to fully
        constrain what will be plotted, the values of the best-performing
        neural network will be used.

        Parameters
        ----------

        y_axis : {'accuracy', 'F1', 'time_taken'}
            Name of performance measure to plot on the y-axis. 'time_taken' is
            the training time *per epoch*.

        vals
            Fixed values to use when taking slice.
        '''

        coords = list(n_argmax(self.accuracy_all, 1)[0])

        # If any coords were automatically set, print out info
        vals = self._autoset_vals(
            [],
            vals,
            coords,
            verbose=True
        )

        max_epoch = 0

        # Set up multi-line plotting
        lines_dims, line_idx_groups, line_val_groups = self._multiline(vals)
        labels = self._line_setup(vals, line_val_groups, classes)
        colors = self._get_palette(vals, classes)

        ########################
        #  Main plotting loop  #
        ########################
        for line_idxs in line_idx_groups:

            # Set coords specific to this line
            for i, idx in enumerate(line_idxs):
                coords[lines_dims[i]] = idx

            if self.success[tuple(coords)]:
                result = self.results[tuple(coords)]

                n_epochs = result['performance']['n_epochs_all']
                max_epoch = max(n_epochs, max_epoch)

                if classes:
                    try:
                        y = [[result['training'][y_axis][i_epoch][self.cls.index(c)]
                              for i_epoch in range(n_epochs)]
                             for c in classes]
                    except KeyError:  # No per-class data for "time" or "loss"
                        err = 'No per-class data for y-axis = ' + str(y_axis)
                        raise KeyError(err)

                    label = labels.next()
                    for i_c, c in enumerate(classes):
                        plt.plot(y[i_c],
                                 label="Class: {}; ".format(c) + label,
                                 color=colors.next()
                                 )

                else:
                    y = result['training'][y_axis + "_all"]
                    plt.plot(y, label=labels.next(), color=colors.next())

            else:
                print 'No data for "' + labels.next() + '"'
                colors.next()

        self._add_bench(plt, y_axis, classes, colors, x_log)
        self._2D_graph_settings(plt, "epoch", y_axis, x_log, (0, max_epoch - 1))
        plt.show()

    def _results_graph(self, x_axis, y_axis, x_log, classes=[], **vals):
        ''' Produce a graph of a hyperparameter against a measure of performance.

        Where the parameters `x_axis`, and `**vals` not been used to fully
        constrain what will be plotted, the values of the best-performing
        neural network will be used.

        Parameters
        ----------

        x_axis : str
            Name of hyperparameter to plot along the x-axis.

        y_axis : {'accuracy', 'F1', 'time_taken'}
            Name of performance measure to plot on the y-axis. 'time_taken' is
            the average training time *per epoch*.

        vals : keyword = {float or str} or {list of float or str} or '*'.
            Additional control of data plotted. Keyword is the name of a
            hyperparameter.
        '''

        # Fix all coordinates at best by default.
        coords = list(n_argmax(self.accuracy_all, 1)[0])

        # Update vals dict to contain autoset values; also print warning
        vals = self._autoset_vals(
            [x_axis],
            vals,
            coords,
            verbose=True
        )

        x_axis_dim = self._dim_names.index(x_axis)
        x_master = self._dim_vals[x_axis_dim]

        # Deal with categorical (non-numeric) x axis
        try:
            x_master[0] - 1.0  # Trigger TypeError on categoricals
            categories = []
            marker = "+"
            ls = '-'
            border = 0.0
        except TypeError:
            categories = list(x_master)
            x_master = range(len(categories))
            plt.xticks(x_master, categories)
            marker = 'o'
            ls = ':'
            border = 0.5

        # Set up a line for all combinations in vals
        lines_dims, line_idx_groups, line_val_groups = self._multiline(vals)
        labels = self._line_setup(vals, line_val_groups, classes)

        colors = self._get_palette(vals, classes)

        # Plotting loop
        for line_idxs in line_idx_groups:

            x = []
            y = [[] for i in range(len(classes))]

            # Set coords specific to this line
            for i, idx in enumerate(line_idxs):
                coords[lines_dims[i]] = idx

            # Step along coords of the x-axis and set y values.
            for xaxis_idx in range(len(x_master)):
                coords[x_axis_dim] = xaxis_idx

                if self.success[tuple(coords)]:
                    result = self.results[tuple(coords)]

                    # Record either overall values or per-class
                    if classes:
                        for i_c, c in enumerate(classes):
                            try:
                                y[i_c].append(result['performance'][
                                              y_axis][self.cls.index(c)])
                            except KeyError:  # Time/n_epochs aren't per-class.
                                err = 'No per-class data for y-axis = ' + \
                                    str(y_axis)
                                raise KeyError(err)
                    else:
                        y.append(result['performance'][y_axis + "_all"])
                    x.append(x_master[xaxis_idx])

            # Plot each class
            if classes and y[0]:
                label = labels.next()
                for i_c, c in enumerate(classes):
                    plt.plot(x, y[i_c],
                             marker=marker,
                             linestyle=ls,
                             label="Class: {}; ".format(c) + label,
                             color=colors.next())

            # Plot one line -- overall results.
            elif not classes and y:
                plt.plot(x, y, marker=marker,
                         linestyle=ls, label=labels.next(), color=colors.next())
            else:
                print 'No data for "' + labels.next() + '"'
                colors.next()

        self._add_bench(plt, y_axis, classes, colors, x_log)
        xlims = (x_master[0] - border, x_master[-1] + border)
        self._2D_graph_settings(plt, x_axis, y_axis, x_log, xlims)

        plt.show()

    def _results_graph3D(self, x_axis, y_axis, x_log, y_log, z_axis='accuracy', **vals):
        '''
        '''

        # Everything except x_axis needs a fixed coordinate
        coords = list(n_argmax(self.accuracy_all, 1)[0])

        # Update vals dict to contain autoset values; also print warning
        vals = self._autoset_vals(
            [x_axis] + [y_axis],
            vals,
            coords,
            verbose=True
        )

        x_axis_dim = self._dim_names.index(x_axis)
        x = self._dim_vals[x_axis_dim]

        y_axis_dim = self._dim_names.index(y_axis)
        y = self._dim_vals[y_axis_dim]

        try:
            x_mesh, y_mesh = np.meshgrid(x, y)
        except TypeError:
            errstr = 'Cannot plot 3D graphs with categorical hyperparameters'
            raise TypeError(errstr)

        z_mesh = np.empty_like(y_mesh, dtype=float)

        # Should return iterators with one value.
        lines_dims, line_idx_groups, _ = self._multiline(vals)
        line_idxs = line_idx_groups.next()

        # Set specified coords
        for i, idx in enumerate(line_idxs):
            coords[lines_dims[i]] = idx

        for i in range(len(x)):
            for j in range(len(y)):
                coords[x_axis_dim] = i
                coords[y_axis_dim] = j

                result = self.results[tuple(coords)]
                z_mesh[j][i] = result['performance'][z_axis + '_all']

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        ax.plot_surface(x_mesh, y_mesh, z_mesh,
                        cmap=cmap, rstride=1, cstride=1)
        ax.set_xlabel(self.xlabels[x_axis])
        ax.set_ylabel(self.xlabels[y_axis])
        ax.set_zlabel(self.results_ylabels[z_axis])

        if x_log:
            ax.xaxis.set_scale('log')

        if y_log:
            ax.yaxis.set_scale('log')

        plt.show()

    def _autoset_vals(self, axes, vals, coords, verbose=False):
        ''' Provide warning that a loose parameter was automatically fixed '''

        manually_set = axes + vals.keys()

        auto_set_dim_names = [dim_name for dim_name in self._dim_names
                              if dim_name not in manually_set]
        auto_set_dims = [self._dim_names.index(dim_name) for dim_name in
                         auto_set_dim_names]
        auto_set_dim_vals = [self._dim_vals[dim][coords[dim]] for dim in
                             auto_set_dims]

        if auto_set_dim_vals:
            warnstr = "Unspecified parameters were automatically fixed:\n"

            for name, value in zip(auto_set_dim_names, auto_set_dim_vals):
                warnstr += "\t" + name + ": " + str(value) + "\n"
                vals.update({name: [value]})

            if verbose:
                warnings.warn(warnstr)

        return vals

    def _2D_graph_settings(self, plt, x_label, y_label, x_log, xlims):

        plt.xlabel(self.xlabels[x_label])

        if x_label == 'epoch':
            plt.ylabel(self.training_ylabels[y_label])
        else:
            plt.ylabel(self.results_ylabels[y_label])

        if x_log:
            plt.xscale('log')

        plt.legend(loc='best')
        plt.xlim(*xlims)

        plt.tight_layout()

    def _add_bench(self, plt, y_axis, classes, colors, x_log):

        lo, hi = -sys.maxint, sys.maxint
        if x_log:
            lo = sys.float_info.min

        if self.MLP.benchmark[y_axis + "_all"] != 0:

            if classes:
                for c in classes:
                    bench = self.MLP.benchmark[y_axis][self.cls.index(c)]
                    plt.plot([lo, hi], [bench, bench],
                             '--', label="[Benchmark] Class: {}".format(c),
                             color=colors.next())

            else:
                bench = self.MLP.benchmark[y_axis + "_all"]
                plt.plot([lo, hi], [bench, bench],
                         linestyle='--', label="Sensible settings (Scikit-learn)",
                         color=colors.next()
                         )

    def _multiline(self, vals):
        ''' Set up iterables for plotting multiple lines on one xy graph.

        Returns
        -------

        lines_dims : list of int
            List of indices in the coords array and all results arrays over
            which the quantities in `vals` vary.

        line_idx_groups : iterable
            Generator of tuples giving a set of indices to use as coordinates
            for the hyperparameters specified to vary in `vals`. Together, the
            tuples list every possible combination.

        line_val_groups : iterable
            Values corresponding to the coordinate sets in `line_idx_groups`
        '''

        dims = [self._dim_names.index(valname) for valname in vals]

        line_val_ars = [list(thisvals) for thisvals in vals.values()]
        line_idx_ars = [[ar_idx(self._dim_vals[dim], val)
                         for val in thisvals]
                        for dim, thisvals in zip(dims, vals.values())]

        line_idx_groups = itertools.product(*line_idx_ars)
        line_val_groups = itertools.product(*line_val_ars)

        return dims, line_idx_groups, line_val_groups

    @staticmethod
    def _get_palette(vals, classes):
        try:
            n_lines = reduce(lambda x, y: x * y,
                             [len(val) for val in vals.values()])
        except TypeError:
            n_lines = 1

        if classes:
            n_lines *= len(classes)

        palette = sns.color_palette("husl", n_lines)

        i = 0

        while True:
            i += 1
            j = (i % n_lines)
            yield palette[j]

    @staticmethod
    def _line_setup(vals, val_groups, classes):
        ''' Produce label differentiating lines in a 2D graph. '''
        ranged_vals = [key for key, val in vals.items() if len(val) > 1]

        for hyper_vals in val_groups:

            label = ""

            for name, val in zip(vals, hyper_vals):
                if name in ranged_vals:
                    label += "{0}: {1}; ".format(name, val)

            if label:
                label = label[:-2]
            else:
                label = 'MLP'

            yield label

    @staticmethod
    def _show_errs(errs):

        if errs:
            print
            print "Some tests failed due to incompatible settings:"
            print "{0:7} | {1:5} | {2}".format("Module", "Count", "Error message")
            for key, count, in errs.items():
                print "{0:7} | {1:5} | {2}".format(key[0], count, key[1])

    @staticmethod
    def _get_static_hyperparams(hypers, search_hypers):
        static_hyperparams = {key: item for (key, item) in hypers.items()
                              if key not in search_hypers}
        return static_hyperparams

    def summary(self, n_models=3, classes=False):
        ''' Summarise the HyperSearch instance.

        Parameters
        ----------

        classes : bool
            Print extra information stratifying performance by class.
        '''

        print
        print "HyperSearch summary"
        print "-------------------"
        print

        dim_len = max_print_len(self._dim_names, 14)
        fmt_prefix = "{:" + dim_len + "} | "

        if not self.search:
            print "Fresh HyperSearch instance. No search parameters given yet."
        else:
            print "Configured hyperparameter search:"

            print (fmt_prefix + "{:>3} | Range").format("Hyperparameter", "N",
                                                         "High val", "Low val")
            for name, values, in zip(self._dim_names, self._dim_vals):
                line = (fmt_prefix + "{:>3} | ").format(name, len(values))
                try:
                    values[0] - 1.0  # Fail if categorical data.
                    line += "{:.3} to {:.3}".format(values[0], values[-1])
                except TypeError:
                    line += " ".join(values)
                except ValueError:  # Probably doesn't want .precision
                    line += "{:3} to {:3}".format(values[0], values[-1])
                finally:
                    print line

            print "{0:22} = {1:>2} models".format(" ", np.prod(self._dims))

            self._show_errs(self.errs)

            print

            if self.ran.all():
                self._show_perf_summary(fmt_prefix, classes=classes, n=n_models)

            else:
                print "The hyperparameter search is incomplete."

        return self

    def _show_perf_summary(self, fmt_prefix, classes=False, n=3):

        try:
            assert(np.prod(self._dims) >= n)
        except AssertionError:
            warnings.warn("Lowering n to number of available models.")
            n = np.prod(self._dims) >= n

        print "Model performance"
        print "-----------------"
        print 

        # Hyperparameters of top models
        print "Top models:"
        coord_sets = n_argmax(self.accuracy_all, n=n)
        print (fmt_prefix + "{:9}").format("Hyperparameter", "Benchmark"),
        for i in range(n):
            print "| {:8}".format("Model " + str(i + 1)),
        print

        for i, (name, values) in enumerate(zip(self._dim_names, self._dim_vals)):
            print (fmt_prefix + "{:>9}").format(name, "-"),
            try:  # Fail if categorical data.
                for j in range(n):
                    print "| {:>8.3g}".format(values[coord_sets[j][i]]),
            except ValueError:
                for j in range(n):
                    print "| {:>8}".format(values[coord_sets[j][i]]),
            finally:
                print

        # Performance measures of top models
        print
        print (fmt_prefix + "{:9}").format("Performance", "Benchmark"),
        for i in range(n):
            print "| {:8}".format("Model " + str(i + 1)),
        print

        specifiers = ['.3f'] * 3 + ['d']

        for perf, spec in zip(self.results_ylabels.keys(), specifiers):
            print (fmt_prefix + "{:9" + spec + "}").\
                format(perf, self.MLP.benchmark[perf + '_all']),
            for j in range(n):
                perf_vals = self.results[coord_sets[j]]['performance']
                print ("| {0:8" + spec + "}").format(perf_vals[perf + "_all"]),
            print

        if classes:
            print
            print "Per-class performance of top model"
            print "----------------------------------"
            print

            # Calculate number of classes to put on one line.
            header2 = "Bench | Model"
            max_chars = 79
            class_len = len(header2)
            prefix_len = len((fmt_prefix[:-2]).format(""))
            cls_line = int(float(max_chars - prefix_len)/class_len)

            remaining_cls = self.n_classes

            while remaining_cls:

                cls_this_line = min(cls_line, remaining_cls)
                i_cls_gen = (range(remaining_cls, remaining_cls - cls_this_line, -1))

                # Table header 1
                print fmt_prefix[:-2].format(""),
                for i_cls in i_cls_gen:
                    cls = self.cls[-i_cls][:class_len]
                    print ("| {:" + str(class_len) + "}").format(cls),
                print

                # Table header 2
                print fmt_prefix[:-2].format("Performance"),
                for _ in i_cls_gen:
                    print ("| " + header2),
                print

                # per-class performance measures
                for perf in self.perclass_ylabels:
                    print fmt_prefix[:-2].format(perf),
                    for i_cls in i_cls_gen:
                        model = self.results[coord_sets[0]]['performance'][perf][- i_cls]
                        bench = self.MLP.benchmark[perf][- i_cls]
                        print "| {0:5.3f} | {1:5.3f}".format(bench, model),
                    print

                remaining_cls -= cls_this_line
                print

    def csv(self, file_name="auto.csv"):
        ''' Output results data as a CSV file for external use.

        The first row provides column labels. Columns are logically grouped as:

        [hyperparameters] [performance] [per-class performance] [training curves]\
        [per-class training curves]

        The second row describes the benchmark classifier. All inapplicable
        fields are set to zero.

        For the series data in the result['training'] dict, the entire series
        is held in a quoted string.
        '''

        hypers = self._dim_names

        results_labels = [key + "_all" for key in self.results_ylabels.keys()]
        training_labels = [key + "_all" for key in self.training_ylabels.keys()]

        training_cls_labels = results_cls_labels = self.perclass_ylabels

        # Strings for naming columns
        cls_results_labels = [key + "_" + cls for key in self.perclass_ylabels
                                            for cls in self.cls]
        cls_training_labels = [key + "_" + cls for key in self.perclass_ylabels
                                            for cls in self.cls]

        coord_gen = itertools.product(*self._dim_idxs)
        val_gen = itertools.product(*self._dim_vals)

        with open(file_name, 'wb') as csvfile:
            fp = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # Write header line
            fp.writerow(hypers + results_labels + cls_results_labels +
                        training_labels + cls_training_labels)

            # Write benchmark line
            vals = [0 for i in range(self._n_dims)]
            performance = [self.MLP.benchmark[label] for label in results_labels]
            class_performance = [self.MLP.benchmark[label][i_cls]
                                    for label in results_cls_labels
                                        for i_cls in range(len(self.cls))]
            training = [0 for label in training_labels]
            class_training = [0 for label in training_cls_labels
                                    for _ in self.cls]

            fp.writerow(list(vals) + performance + class_performance +
                        training + class_training)

            # Write lines
            for coords, vals in zip(coord_gen, val_gen):
                if not self.success[coords]:
                    continue

                result = self.results[coords]
                n_epochs = result['performance']['n_epochs_all']

                performance = [result['performance'][label]
                                        for label in results_labels]
                class_performance = [result['performance'][label][i_cls]
                                        for label in results_cls_labels
                                            for i_cls in range(len(self.cls))]
                training = [joinstr(" ", result['training'][label])
                                        for label in training_labels]
                class_training = [joinstr(" ", [result['training'][label][j_epoch][i_cls]
                                        for j_epoch in range(n_epochs)])
                                            for label in training_cls_labels
                                                for i_cls in range(len(self.cls))]

                fp.writerow(list(vals) + performance + class_performance +
                            training + class_training)






        return None

def joinstr(string, iterable):
    ''' Like the str method join(), but works when the items are not str. '''

    return string.join([str(item) for item in iterable])


def ar_idx(arr, value):
    ''' Implement  list([e1, e2, ...]).index(element) for numpy arrays '''

    try:  # Necessary for float
        idx = np.isclose(arr, value).nonzero()[0][0]
    except TypeError:  # Categorical
        idx = (arr == value).nonzero()[0][0]

    return idx


def max_print_len(elems, min_len=0):
    ''' Return maximum length of elements in list represented as str '''

    max_len = min_len

    for elem in elems:
        max_len = max(max_len, len(str(elem)))

    return str(max_len)


def n_argmax(arr, n=1):
    ''' Find the coordinates of the top n values in arr. '''

    try:
        assert(np.prod(arr.shape) >= n)
    except AssertionError:
        warnings.warn("n requested top values exceeds array size." +
                      " Auto-setting n to the array size.")
        n = np.prod(arr.shape) >= n

    coords = []
    vals = []

    min_val = np.array(arr).min()

    for i in range(n):
        # Find global maximum
        max_coords = np.unravel_index(arr.argmax(), arr.shape)

        # Store coords, values at these coords
        coords.append(max_coords)
        vals.append(arr[max_coords])

        # Set value equal to minimum value so it won't be found next time.
        arr[max_coords] = min_val

    # Reset modified values in arr
    for coord, val in zip(coords, vals):
        arr[coord] = val

    return coords
