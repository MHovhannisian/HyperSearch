#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core Python modules
import pickle
import itertools
import warnings
import collections

# User-contributed modules
import numpy as np
import progressbar

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import unifiedmlp


class HyperSearch(object):

    def __init__(self, X, Y, class_names=[], split=(0.70, 0.15, 0.15)):

        self.n_classes = Y.shape[1]

        if class_names:
            self.class_names = tuple(class_names)
        else:
            self.class_names = tuple([str(i) for i in range(self.n_classes)])

        self.MLP = unifiedmlp.UnifiedMLP(X, Y, split)

        self._dim_names = []
        self._dim_vals = []

        self.fixed = False
        self.search = False

        # Data on outputting
        self.performance = tuple(['F1', 'accuracy', 'time'])
        self.results_ylabels = {'F1': 'F1 score (test data)',
                                'accuracy': 'Accuracy score (test data)',
                                'time': 'Training time per epoch (seconds)'
                                }
        self.training_ylabels = {'F1': 'F1 score (validation data)',
                                 'accuracy': 'Accuracy score (validation data)',
                                 'time': 'Training time for epoch (seconds)',
                                 'loss': 'Loss function (training data)'
                                 }

        self.xlabels = {
            'module': 'Python module',
            'frac_training': 'Fraction training data used',

            'hidden_layer_size': 'Number of hidden units',
            'activation': 'Activation function in hidden layer',

            'alpha': 'L2 penalty (alpha)',
            'dropout': 'Dropout (probability)',

            'learning_rate': 'Initial learning rate',
            'algorithm': 'Learning algorithm',
            'batch_size': 'Samples per minibatch',

            'momentum': 'SGD momentum term',
            'nesterovs_momentum': 'Nesterov\'s momentum enabled',

            'beta_1': 'beta_1 parameter (Adam)',
            'beta_2': 'beta_2 parameter (Adam)',
            'epsilon': 'epsilon parameter (Adam)',

            'max_epoch': 'Maximum epochs before stopping',
            'epoch_tol': 'Tolerance on stopping criteria',

            'n_stable': 'Consecutive stable epochs before stopping',

            'learning_decay': 'Per-epoch learning rate decay (SGD)',

            'early_stopping': 'Performance on validation data as stopping criterion'
        }

        self.errs = collections.Counter()

    def set_fixed(self, **hypers):
        assert(not self.fixed)
        self.fixed = True
        self.MLP.set_hypers(**hypers)

        return self

    def set_search(self, **hypers):
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

        return self

    def run_tests(self, ignore_failure=False):
        ''' Execute the specified parameter search. '''

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
                result, _ = self.MLP.run_test()
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

    def graph(self, x_axis='epoch', y_axis='accuracy', z_axis='accuracy',
              x_log=False, y_log=False, **vals):

        # Setup

        if not self.ran.all():
            print "Cannot make graphs - please run run_tests()."
            return self

        # Homogenise inputs into a list of raw vals.
        # Careful typing here to avoid ambiguities
        for key, item in vals.items():
            if item == '*':
                dim = self._dim_names.index(key)
                vals[key] = list(self._dim_vals[dim])
            elif (isinstance(item, float) or isinstance(item, int)
                    or isinstance(item, basestring)):
                vals[key] = [item]
            elif hasattr(item, "__getitem__"):
                pass
            else:
                errstr = ("Pass hyperparams as either a raw value, a list of"
                          + " raw values, or '*'.")
                print item
                raise TypeError(errstr)

        # Call appropriate graphing function

        if x_axis == 'epoch':
            self._training_graph(y_axis, x_log, **vals)
        elif y_axis not in self.performance:
            try:
                assert(z_axis in self.performance)
            except AssertionError:
                raise AssertionError("For 3D graphs, the performance metric" +
                                     " should be on the z-axis and no seperate lines can be specified")
            self._results_graph3D(x_axis, y_axis, x_log, y_log, z_axis, **vals)
        else:
            self._results_graph(x_axis, y_axis, x_log, **vals)

        return self

    def _training_graph(self, y_axis, x_log, **vals):
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
        specified = vals.keys()
        vals = self._autoset_vals(
            [],
            vals,
            coords,
            verbose=True
        )

        y = []
        ymaxlen = 0

        # Set up multi-line plotting
        lines_dims, line_idx_groups, line_val_groups = self._multiline(vals)
        labels = self._line_label(vals.keys(), line_val_groups, specified)

        # Plotting loop
        for line_idxs in line_idx_groups:

            # Set coords specific to this line
            for i, idx in enumerate(line_idxs):
                coords[lines_dims[i]] = idx

            if self.success[tuple(coords)]:
                y = self.results[tuple(coords)]['training'][y_axis + "_all"]
                ymaxlen = max(len(y), ymaxlen)

            # Plot
            if y:
                plt.plot(y, label=labels.next())
            else:
                print 'No data for "' + labels.next() + '"'

            y = []

        try:
            bench = self.MLP.benchmark[y_axis + "_all"]
            plt.plot([0, ymaxlen - 1], [bench, bench],
                     '-.', label="Stratified Random")
        except KeyError:
            pass

        plt.xlabel("Epoch")
        plt.ylabel(self.training_ylabels[y_axis])

        if x_log:
            plt.xscale('log')

        plt.legend(loc='best')
        plt.show()

    def _results_graph(self, x_axis, y_axis, x_log, **vals):
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
        specified = vals.keys()
        vals = self._autoset_vals(
            [x_axis],
            vals,
            coords,
            verbose=True
        )

        x_axis_dim = self._dim_names.index(x_axis)
        x_master = self._dim_vals[x_axis_dim]
        x, y = [], []

        # Deal with categorical (non-numeric) x axis
        try:
            x_master[0] - 1.0
            categories = []
        except TypeError:
            categories = list(x_master)
            x_master = range(len(categories))

        # Set up a line for all combinations in vals
        lines_dims, line_idx_groups, line_val_groups = self._multiline(vals)
        labels = self._line_label(vals.keys(), line_val_groups, specified)

        # Plotting loop
        for line_idxs in line_idx_groups:

            # Set coords specific to this line
            for i, idx in enumerate(line_idxs):
                coords[lines_dims[i]] = idx

            # Step along coords of the x-axis and set y values.
            for i in range(len(x_master)):
                coords[x_axis_dim] = i
                if self.success[tuple(coords)]:
                    result = self.results[tuple(coords)]
                    y.append(result['performance'][y_axis + "_all"])
                    x.append(x_master[i])

            # Plot
            if y:
                if categories:
                    # Dummy numerical system with dashed lines
                    plt.plot(x, y, 'o--', label=labels.next())
                    plt.xticks(x_master, categories)
                else:
                    plt.plot(x, y, label=labels.next())
            else:
                print 'No data for "' + labels.next() + '"'

            x, y = [], []

        try:
            bench = self.MLP.benchmark[y_axis + "_all"]
            plt.plot([x_master[0], x_master[-1]], [bench, bench],
                     '-.', label="Stratified Random")
        except KeyError:
            pass

        plt.xlabel(self.xlabels[x_axis])
        plt.ylabel(self.results_ylabels[y_axis])

        if x_log:
            plt.xscale('log')

        plt.legend(loc='best')
        plt.show()

    def _results_graph3D(self, x_axis, y_axis, x_log, y_log, z_axis='accuracy', **vals):
        '''
        '''

        # Everything except x_axis needs a fixed coordinate
        coords = list(n_argmax(self.accuracy_all, 1)[0])

        # Update vals dict to contain autoset values; also print warning
        specified = vals.keys()
        vals = self._autoset_vals(
            [x_axis]+[y_axis],
            vals,
            coords,
            verbose=True
        )

        x_axis_dim = self._dim_names.index(x_axis)
        x = self._dim_vals[x_axis_dim]

        y_axis_dim = self._dim_names.index(y_axis)
        y = self._dim_vals[y_axis_dim]
        x_mesh, y_mesh = np.meshgrid(x, y)

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

        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cmap, rstride=1, cstride=1)
        ax.set_xlabel(self.xlabels[x_axis])
        ax.set_ylabel(self.xlabels[y_axis])
        ax.set_zlabel(self.results_ylabels[z_axis])

        if x_log:
            ax.xaxis.set_scale('log')

        if y_log:
            ax.yaxis.set_scale('log')

        plt.show()

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
        line_idx_ars = [[ar_index(self._dim_vals[dim], val)
                         for val in thisvals]
                        for dim, thisvals in zip(dims, vals.values())]

        line_idx_groups = itertools.product(*line_idx_ars)
        line_val_groups = itertools.product(*line_val_ars)

        return dims, line_idx_groups, line_val_groups

    def _get_fixed_coords(self, fixed_vals):
        ''' Returns coordinates for taking slices of the data.

        For viewing slices of the data, selects the fixed values of the
        hyperparameters orthogonal to the slice. Selects the coordinates of the
        maximum-accuracy hyperparameter combination, then overwrites this
        with any constraints provided by user-provided fixed_vals dict.

        Parameters
        ----------

        fixed_vals : dict
            Specifies constraints on the fixed value.

            e.g. {"dropout": 0.5} to take the slice where "dropout" is held
            constant at 0.5.

        Returns
        -------

        fixed_coords : list
            Selected co-ordinates. Functions using this are expected to
            overwrite the co-ordinates in the direction of the slice.

        '''

        # Obtain coordinates of the global accuracy maximum

        return fixed_coords

    @staticmethod
    def _line_label(hyper_names, hyper_val_sets, specified):
        ''' Produce label differentiating lines in a 2D graph. '''

        for hyper_vals in hyper_val_sets:
            label = ""

            for name, val in zip(hyper_names, hyper_vals):
                if name in specified:
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

    def summary(self, class_split=False):
        ''' Summarise the HyperSearch instance. '''

        print
        print "HyperSearch summary"
        print "-------------------"
        print

        if not self.search:
            print "Fresh HyperSearch instance. No search parameters given yet."
        else:
            print "Configured hyperparameter search:"

            print "{0:14} | {1:3} | Range".format("Hyperparameter", "N", "High val", "Low val")
            for name, values, in zip(self._dim_names, self._dim_vals):
                line = "{0:14} | {1:3} | ".format(name, len(values))
                try:
                    values[0] - 1.0  # Fail if categorical data.
                    line += "{:.3} to {:.3}".format(values[0], values[-1])
                except TypeError:
                    line += " ".join(values)
                except ValueError:  # Probably doesn't want .precision
                    line += "{:3} to {:3}".format(values[0], values[-1])
                finally:
                    print line

            print "{0:19} = {1:>2} models".format(" ", np.prod(self._dims))

            self._show_errs(self.errs)

            print

            if self.ran.all():
                self._show_perf_summary(class_split=class_split)

            else:
                print "The hyperparameter search is incomplete."

        return self

    def _show_perf_summary(self, class_split=False, n=3):

        try:
            assert(np.prod(self._dims) >= n)
        except AssertionError:
            warnings.warn("Lowering n to number of available models.")
            n = np.prod(self._dims) >= n

        # Hyperparameters of top models
        print "Top models:"
        coord_sets = n_argmax(self.accuracy_all, n=n)
        print "{0:14} | {1:9}".format("Hyperparameter", "Benchmark"),
        for i in range(n):
            print "| {:7}".format("Model " + str(i + 1)),
        print

        for i, (name, values) in enumerate(zip(self._dim_names, self._dim_vals)):
            print "{0:14} | {1:>9}".format(name, "-"),
            try:  # Fail if categorical data.
                for j in range(n):
                    print "| {:>7.3g}".format(values[coord_sets[j][i]]),
            except ValueError:
                for j in range(n):
                    print "| {:>7}".format(values[coord_sets[j][i]]),
            finally:
                print

        # Performance measures of top models
        print
        print "{0:14} | {1:9}".format("Performance", "Benchmark"),
        for i in range(n):
            print "| {:7}".format("Model " + str(i + 1)),
        print
        for perf in self.performance:
            print "{0:14} | {1:9.3}".format(perf, self.MLP.benchmark[perf + '_all']),
            for j in range(n):
                perf_vals = self.results[coord_sets[j]]['performance']
                print ("| {0:7.3f}".  format(perf_vals[perf + "_all"])),
            print


def ar_index(arr, value):

    try:  # Works for non-categorical
        idx = np.isclose(arr, value).nonzero()[0][0]
    except TypeError:
        idx = (arr == value).nonzero()[0][0]

    return idx


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

    minval = np.array(arr).min()

    for i in range(n):
        # Find global maximum
        max_coords = np.unravel_index(arr.argmax(), arr.shape)

        # Store coords, values at these coords
        coords.append(max_coords)
        vals.append(arr[max_coords])

        # Set value equal to minimum value so it won't be found next time.
        arr[max_coords] = minval

    # Reset modified values in arr
    for coord, val in zip(coords, vals):
        arr[coord] = val

    return coords
