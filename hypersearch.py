#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core Python modules
import pickle
import itertools
import warnings
from collections import Counter

# User-contributed modules
import numpy as np
import progressbar

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import unifiedmlp

class HyperSearch(object):

    def __init__(self, X, Y, split=(0.70, 0.15, 0.15)):

        self.MLP = unifiedmlp.UnifiedMLP(X, Y, split)

        self._dim_names = []
        self._dim_vals = []

        self.fixed = False
        self.search = False

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
        self._dims = np.array([len(self._dim_vals[i]) for i in range(self._n_dims)])
        self._dim_idxs = tuple(range(len(self._dim_vals[i])) for i in range(self._n_dims))

        self.results = np.empty(self._dims, dtype=object)
        self.ran = np.zeros(self._dims, dtype=bool)
        self.success = np.ones(self._dims, dtype=bool)

        self.accuracy = np.zeros(self._dims, dtype=float)
        self.F1 = np.zeros(self._dims, dtype=float)

        self.performance = ['F1', 'accuracy', 'time_taken']

        return self

    def run_tests(self, ignore_failure=False):
        ''' Execute the specified parameter search. '''

        try:
            assert (self.search)
        except AssertionError:
            raise AssertionError('No parameters to search over were specified.')

        err_count = Counter()
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

            try:
                result, _ = self.MLP.run_test()
                self.success[coordinates] = True

                self.results[coordinates] = result
                self.accuracy[coordinates] = result['performance']['accuracy']
                self.F1[coordinates] = result['performance']['F1']

            except KeyError as e: # When unsupported settings are specified
                if ignore_failure:
                    self.success[coordinates] = False
                    err = (this_settings['module'], e.message)
                    err_count[err] += 1
                else:
                    raise
            finally:
                self.ran[coordinates] = True

            bar.update(i+1)

        bar.finish()

        if err_count:
            print
            warnings.warn('Some tests failed due to incompatible settings:')
            print "{0:7} | {1:5} | {2}".format("Module", "Count", "Error message")
            for key, count, in err_count.items():
                print "{0:7} | {1:5} | {2}".format(key[0], count, key[1])

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

    def _warn_autoset_coords(self, manually_set, coords):
        ''' Provide warning that a loose parameter was automatically fixed '''

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
            warnings.warn(warnstr)

    def _results_graph3D(self, x_axis, y_axis, z_axis='accuracy', **vals):
        '''
        '''

        # Everything except x_axis needs a fixed coordinate
        coords = self._get_fixed_coords(vals)

        # If any coords were automatically set, print out info
        self._warn_autoset_coords([x_axis]+vals.keys()+[y_axis], coords)

        x_axis_dim = self._dim_names.index(x_axis)
        x = self._dim_vals[x_axis_dim]

        y_axis_dim = self._dim_names.index(y_axis)
        y = self._dim_vals[y_axis_dim]
        x_mesh, y_mesh = np.meshgrid(x, y)

        z = np.empty_like(y_mesh)

        for i in range(len(x)):
            for j in range(len(y)):
                coords[x_axis_dim] = i
                coords[y_axis_dim] = j

                result = self.results[tuple(coords)]
                z[j][i] = result['performance'][z_axis]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        ax.plot_surface(x_mesh, y_mesh, z, cmap=cmap, rstride=1, cstride=1)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)

        plt.show()

    def _multiline2D(self, lines):
        ''' Set up iterables for plotting multiple lines on one xy graph.

        Returns
        -------

        lines_dims : list of int
            List of indices in the coords array and all results arrays over
            which the quantities in `lines` varies.

        line_group_idxs : iterable
            Generator of tuples giving a set of indices to use as coordinates
            for the hyperparameters specified to vary in `lines`. Together, the
            tuples list every possible combination.

        line_group_vals : iterable
            Values corresponding to the coordinate sets in `line_group_idxs`
        '''

        if lines:
            lines_dims = [self._dim_names.index(line) for line in lines]
            line_idx_ars = [self._dim_idxs[lines_dim] for lines_dim in lines_dims]
            line_val_ars = [self._dim_vals[lines_dim] for lines_dim in lines_dims]
            line_group_idxs = itertools.product(*line_idx_ars)
            line_group_vals = itertools.product(*line_val_ars)
        else:
            # Dummy val
            line_group_idxs = [(0,)]
            line_group_vals = iter([0])
            lines_dims = None

        return lines_dims, line_group_idxs, line_group_vals

    def _training_graph(self, y_axis='accuracy', lines=[], **vals):
        ''' Graph test/validation performance *per epoch*.

        Where the parameters `lines`, and `**vals` not been used to fully
        constrain what will be plotted, the values of the best-performing
        neural network will be used.

        Parameters
        ----------

        y_axis : {'accuracy', 'F1', 'time_taken'}
            Name of performance measure to plot on the y-axis. 'time_taken' is
            the training time *per epoch*.

        lines : list of str
            Seperate lines will be plotted for each value of this hyperparameter.

        vals
            Fixed values to use when taking slice.
        '''

        coords = self._get_fixed_coords(vals)

        # If any coords were automatically set, print out info
        self._warn_autoset_coords(vals.keys()+lines, coords)

        y = []
        ymaxlen = 0

        # Set up multi-line plotting
        lines_dims, line_group_idxs, line_group_vals = self._multiline2D(lines)

        # Plotting loop
        for line_group_idx in line_group_idxs:

            # Set coords specific to this line
            if lines:
                for i_idx, idx in enumerate(line_group_idx):
                    coords[lines_dims[i_idx]] = idx

            if self.success[tuple(coords)]:
                y = self.results[tuple(coords)]['training'][y_axis]
                ymaxlen = max(len(y), ymaxlen)

            label = self._line_label(lines, line_group_vals.next())

            # Plot
            if y:
                plt.plot(y, label=label)

            y = []

        bench = self.MLP.benchmark[y_axis]
        plt.plot([0, ymaxlen - 1], [bench, bench], '-.', label="Stratified Random")
        plt.xlabel("Epoch")
        plt.ylabel(y_axis)

        plt.legend(loc='best')
        plt.show()

    def graph(self, x_axis='epoch', y_axis='accuracy', z_axis=None,
            lines=[], **vals):

        if not self.ran.all():
            print "Cannot make graphs - please run run_tests()."
            return self

        if x_axis == 'epoch':
            self._training_graph(y_axis, lines, **vals)
        elif z_axis:
            try:
                assert(not lines and y_axis not in performance)
            except AssertionError:
                raise("For 3D graphs, the performance metric should be on the"
                        + " z-axis and no seperate lines can be specified")
            self._results_graph3D(x_axis, y_axis, z_axis, **vals)
        else:
            self._results_graph(x_axis, y_axis, lines, **vals)

        return self

    def _results_graph(self, x_axis, y_axis='accuracy', lines=[], **vals):
        ''' Produce a graph of a hyperparameter against a measure of performance.

        Where the parameters `lines`, and `**vals` not been used to fully
        constrain what will be plotted, the values of the best-performing
        neural network will be used.

        Parameters
        ----------

        x_axis : str
            Name of hyperparameter to plot along the x-axis.

        y_axis : {'accuracy', 'F1', 'time_taken'}
            Name of performance measure to plot on the y-axis. 'time_taken' is
            the average training time *per epoch*.

        lines : list of str
            Seperate lines will be plotted for each value of this hyperparameter.

        vals
            Fixed values to use when taking slice.
        '''

        # Everything except x_axis needs a fixed coordinate
        coords = self._get_fixed_coords(vals)

        # If any coords were automatically set, print out info
        self._warn_autoset_coords([x_axis]+vals.keys()+lines, coords)

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

        # Set up multi-line plotting
        lines_dims, line_group_idxs, line_group_vals = self._multiline2D(lines)

        # Plotting loop
        for line_group_idx in line_group_idxs:

            # Set coords specific to this line
            if lines:
                for i_idx, idx in enumerate(line_group_idx):
                    coords[lines_dims[i_idx]] = idx

            # Step along coords of the x-axis and set y values.
            for i in range(len(x_master)):
                coords[x_axis_dim] = i
                if self.success[tuple(coords)]:
                    result = self.results[tuple(coords)]
                    y.append(result['performance'][y_axis])
                    x.append(x_master[i])

            label = self._line_label(lines, line_group_vals.next())

            # Plot
            if y:
                if categories:
                    # Dummy numerical system with dashed lines
                    plt.plot(x, y, 'o--', label=label)
                    plt.xticks(x_master, categories)
                else:
                    plt.plot(x, y, label=label)

            x, y = [], []

        bench = self.MLP.benchmark[y_axis]
        plt.plot([x_master[0],x_master[-1]], [bench, bench], '-.', label="Stratified Random")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        plt.legend(loc='best')
        plt.show()

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
        max_coords = np.unravel_index(self.accuracy.argmax(), self.accuracy.shape)

        # Set return value to these coordinates
        fixed_coords = [max_coords[i] for i in range(self._n_dims)]

        for key, value in fixed_vals.items():
            dim = self._dim_names.index(key)
            try: # Works for non-categorical
                fixed_coords[dim] = np.isclose(self._dim_vals[dim], value).nonzero()[0][0]
            except TypeError:
                fixed_coords[dim] = (self._dim_vals[dim] == value).nonzero()[0][0]

        return fixed_coords

    @staticmethod
    def _line_label(hyperparam_names, hyperparam_values):
        ''' Produce label differentiating lines in a 2D graph. '''

        if hyperparam_names:
            label = "{0}: {1}".format(hyperparam_names[0], hyperparam_values[0])

            for i in range(1, len(hyperparam_names)):
                label += "; {0}: {1}".format(hyperparam_names[i], hyperparam_values[i])
        else:
            label = 'MLP'

        return label


if __name__ == "__main__":
    # data_file = "data_top5.pickle"
    # with open(data_file, 'r') as datafile:
        # pd_df = pickle.load(datafile)

    # X = pd_df.ix[:, 0:13]
    # Ys = pd_df.ix[:, 15:25]

    # hs = HyperSearch(X, Ys)
    # hs.set_fixed(max_epoch=10)
    # hs.set_search(module=['sklearn', 'keras', 'sknn'],
                  # algorithm=['adam','adadelta', 'sgd'],
                  # dropout=[0.0, 0.5],
                  # frac_training=np.arange(0.1, 1.1, 0.1)
                 # )
    # hs.run_tests(ignore_failure=True).save()

    hs = HyperSearch.load('Angie.save')
    hs.graph(x_axis='frac_training', lines=['algorithm', 'dropout'], module='keras')
    # hs.graph(x_axis='dropout')
