#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core Python modules
import pickle
import itertools

# User-contributed modules
import numpy as np
import progressbar

import unifiedmlp

class HyperSearch(object):

    def __init__(self, X, Y, split=(0.70, 0.15, 0.15),
            autosave="HyperSearch.pickle"):

        self.autosave = autosave

        self.MLP = unifiedmlp.UnifiedMLP(X, Y, split)

        self._search_hyper_names = []
        self._search_hyper_vals = []

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
            self._search_hyper_names.append(key)
            self._search_hyper_vals.append(np.sort(np.array(range_)))

        self._n_dims = len(self._search_hyper_names)
        self._dims = np.array([len(self._search_hyper_vals[i]) for i in range(self._n_dims)])
        self._dim_idxs = tuple(range(len(self._search_hyper_vals[i])) for i in range(self._n_dims))

        self.tests = np.empty(self._dims, dtype=object)

        return self

    def run_tests(self):
        try:
            assert (self.search)
        except AssertionError:
            raise AssertionError('No parameters to search over were specified.')

        n_tests = np.prod(self._dims)
        bar = progressbar.ProgressBar(maxval=n_tests).start()

        for i, coordinates in enumerate(itertools.product(*self._dim_idxs)):
            this_settings = {self._search_hyper_names[i]: self._search_hyper_vals[i][coord]
                                    for i, coord in enumerate(coordinates)}
            self.MLP.set_hypers(**this_settings)

            self.tests[coordinates] = self.MLP.run_test()

            bar.update(i+1)

    def self_pickle(self):
        ''' Save the state of the HyperSearch instance. '''

        with open(self.autosave, 'w') as pickle_file:
            pickle.dump(self, pickle_file)

        return self

    @staticmethod
    def unpickle(name="HyperSearch.pickle"):
        ''' Load a previously saved HyperSearch instance. '''

        with open(name, 'r') as pickle_file:
            instance = pickle.load(pickle_file)

        return instance

if __name__ == "__main__":
    data_file = "data_top5.pickle"
    with open(data_file, 'r') as datafile:
        pd_df = pickle.load(datafile)

    X = pd_df.ix[:, 0:13]
    Ys = pd_df.ix[:, 15:25]

    hs = HyperSearch(X, Ys).self_pickle().set_fixed(hi="bye")
    hs.set_search(dropout=[0.0, 0.1], alpha=[0.0, 0.0])
    hs.run_tests()
