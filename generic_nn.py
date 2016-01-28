#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Module containing a class which allows access to disparate Python neural
network implementations and architectures, united through a common interface.
This interface is modelled on the scikit-learn interface.
'''

from sklearn.cross_validation import KFold
from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier as sklearn_MLPClassifier

from sknn.mlp import Classifier as sknn_MLPClassifier, Layer

from pybrain.structure import FeedForwardNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets            import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import numpy as np

class NN_Compare:
    ''' Class allows training and testing multiple neural network architectures
        from disparate Python models on the same data and conveniently
        outputting the results.

        Works with 1 hidden layer for now.
    '''

    def __init__(self, X, Y, k_folds=4):
        self.X = np.array(X).astype('float64')
        self.Y = np.array(Y).astype(bool)

        try:
            assert(self.X.shape[0] == self.Y.shape[0])
        except AssertionError:
            print "ERROR: Number of samples differs between X and Y!"
            exit()

        self.n_samples = self.Y.shape[0]
        self.n_features = self.X.shape[1]
        self.n_outcomes = self.Y.shape[1]

        self.settings = {
            'hidden_layer_sizes' : (13,)
        }

        self.k_folds = k_folds
        self.k_fold = KFold(self.n_samples, n_folds=k_folds, shuffle=True,
                            random_state=1)

    def get_settings(self):
        return self.settings

    def set_settings(self, new_settings):
        self.settings.update(new_settings)
        return self

    def run_test(self, nntype='sklearn_MLP'):
        nntypes = {
            'sklearn_MLP' : self.sklearn_MLP,
            'sknn_MLP'    : self.sknn_MLP,
            'pybrain_MLP' : self.pybrain_MLP,
            'keras_MLP'    : self.keras_MLP
        }

        F_score = 0.0
        percent_score = 0.0

        for train_idx, test_idx in self.k_fold:
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]

            # Train the normaliser on training data only (to not cheat)
            normaliser = preprocessing.StandardScaler().fit(X_train)
            X_train = normaliser.transform(X_train)
            X_test = normaliser.transform(X_test)

            Y_test_predicted = nntypes[nntype](X_train, Y_train, X_test)

            # score = self.get_score(Y_test_predicted, Y_test)
            # F_score += score[0]/self.k_folds
            # percent_score += score[1]/self.k_folds
            break

        # return F_score, percent_score
        return Y_test_predicted

    def keras_MLP(self, X_train, Y_train, X_test):
        nn = Sequential()

        nn.add(Dense(self.settings['hidden_layer_sizes'][0],
                        input_dim=self.n_features,
                        init='uniform')
                 )
        nn.add(Activation('linear'))
        nn.add(Dropout(0.5))

        nn.add(Dense(self.n_outcomes, init='uniform'))
        nn.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        nn.compile(loss='mean_squared_error', optimizer=sgd)

        nn.fit(X_train, Y_train, nb_epoch=1, batch_size=16)
        plot(nn, to_file='model.png')

        return nn.predict_proba(X_test)

    def sklearn_MLP(self, X_train, Y_train, X_test):
        nn = sklearn_MLPClassifier(**self.settings)
        nn.fit(X_train, Y_train)
        print nn
        Y_test_predicted = nn.predict_proba(X_test)
        return Y_test_predicted

    def sknn_MLP(self, X_train, Y_train, X_test):
        nn = sknn_MLPClassifier(
                layers=[Layer("Rectifier", units=self.settings['hidden_layer_sizes'][0]),
                        Layer("Sigmoid")],
                verbose=True
        )

        nn.fit(X_train, Y_train)
        Y_test_predicted = nn.predict(X_test)
        return Y_test_predicted

    def pybrain_MLP(self, X_train, Y_train, X_test):
        # nn = buildNetwork(self.n_features, self.settings['hidden_layer_sizes'][0],
                          # self.n_outcomes, outclass=SoftmaxLayer)
        nn = FeedForwardNetwork()

        inLayer = LinearLayer(self.n_features)
        hiddenLayer = LinearLayer(self.settings['hidden_layer_sizes'][0])
        outLayer = SigmoidLayer(self.n_outcomes)

        nn.addInputModule(inLayer)
        nn.addModule(hiddenLayer)
        nn.addOutputModule(outLayer)

        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)

        nn.addConnection(in_to_hidden)
        nn.addConnection(hidden_to_out)
        nn.sortModules()

        train = ClassificationDataSet(inp=self.n_features, target=self.n_outcomes)
        train.setField('input', X_train)
        train.setField('target', Y_train)

        test = ClassificationDataSet(inp=self.n_features, target=self.n_outcomes)
        # NOTE using dummy Y_test var. Should work and seems to work.
        Y_test = np.ones([X_test.shape[0], self.n_outcomes])
        test.setField('input', X_test)
        test.setField('target', Y_test)

        trainer = BackpropTrainer(nn, dataset=train, momentum=0.1, verbose=True, weightdecay=0.01)
        trainer.trainEpochs(1)

        out = nn.activateOnDataset(test)
        return out

    def get_score(self, predictions, answers):
        ''' Returns the F1 score and simple score (percent correct).
            Requires predictions and answers in 0 and 1 int or bool format.
        '''
        predicted_positives = (predictions == 1)
        print predicted_positives
        print answers
        true_positives = (predicted_positives & answers)
        false_positives = (predicted_positives & np.logical_not(answers))
        correct_predictions = (predictions == answers)

        precision = float(true_positives.sum())/predicted_positives.sum()
        recall = float(true_positives.sum())/answers.sum()
        F1 = (2*precision*recall)/(precision+recall)
        score = float(correct_predictions.sum())/(predictions.shape[0]*predictions.shape[1])

        return F1, score
