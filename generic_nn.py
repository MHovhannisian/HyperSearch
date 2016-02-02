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
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection

import numpy as np
# NOTE: This is the only simple way to get reproducable results in Keras.
np.random.seed(1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.regularizers import l2

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
        self.kfold_seed = 1

        # Settings which are natively understood by scikit-learn
        # http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPClassifier.html
        self.settings = {
            # Architecture
            'hidden_layer_sizes' : (15,),
            'activation' : 'relu',

            # Regularisation
            'alpha' : 0.0002, # L2 penalty. 0.0 = turned off.

            # Learning
            'learning_rate_init' : 0.001,
            'algorithm' : 'sgd',
            # SGD only
            'momentum' : 0.9,
            'nesterovs_momentum' : True,

            # Consistent output (for developing and debugging)
            # Doesn't work in Keras (there's a hack though -- see the imports)
            # Doesn't work in PyBrain and no known hack :(
            'random_state' : 1
        }

        # For settings which take a categorical value, provided is a dict of
        # which settings should work in which of the Python modules.
        # This dict exists only for reference. It is not used for computaton.
        self.supported_settings = {
            'activation' : {
                'relu' : ['sklearn', 'sknn', 'keras'],
                'linear' : ['sknn', 'pybrain', 'keras'],
                'logistic' : ['sklearn', 'sknn', 'pybrain', 'keras'],
                'tanh' : ['sklearn', 'sknn', 'pybrain', 'keras']
            }
        }

        self.k_folds = k_folds
        self.k_fold = KFold(self.n_samples, n_folds=k_folds, shuffle=True,
                            random_state=self.kfold_seed)

    def get_settings(self):
        return self.settings

    def set_settings(self, new_settings):
        self.settings.update(new_settings)
        return self

    def run_test(self, nntype):
        nntypes = {
            'sklearn' : self.sklearn,
            'sknn'    : self.sknn,
            'pybrain' : self.pybrain,
            'keras'    : self.keras
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

    def keras(self, X_train, Y_train, X_test):
        # Settings conversion
        activation_dict = {'relu': 'relu', 'linear': 'linear', 'logistic': 'sigmoid', 'tanh': 'tanh'}
        try:
            activation = activation_dict[self.settings['activation']]
        except KeyError:
            print "ERROR: Activation function \"" + self.settings['activation'] + "\"",
            print "not supported in Keras."
            raise NotImplementedError

        # Create nn architecture
        nn = Sequential()

        nn.add(Dense(
            self.settings['hidden_layer_sizes'][0],
            input_dim=self.n_features,
            init='uniform',
            activation=activation)
        )
        # nn.add(Dropout(0.5))

        nn.add(Dense(
            self.n_outcomes,
            init='uniform',
            activation='softmax')
        )

        print self.settings

        if self.settings['algorithm'] == 'sgd':
            sgd = SGD(
                lr=self.settings['learning_rate_init'],
                decay=1e-6,
                momentum=self.settings['momentum'],
                nesterov=self.settings['nesterovs_momentum'],
            )
        else:
            print "ERROR: Only SGD is implemented in Keras at present."
            raise NotImplementedError

        nn.compile(loss='mean_squared_error', optimizer=sgd)

        nn.fit(X_train, Y_train, nb_epoch=1, batch_size=16)

        return nn.predict_proba(X_test)

    def sklearn(self, X_train, Y_train, X_test):
        nn = sklearn_MLPClassifier(**self.settings)
        print nn
        nn.fit(X_train, Y_train)
        Y_test_predicted = nn.predict_proba(X_test)
        return Y_test_predicted

    def sknn(self, X_train, Y_train, X_test):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {'relu': 'Rectifier', 'linear': 'Linear', 'logistic': 'Sigmoid', 'tanh': 'Tanh'}
        try:
            activation = activation_dict[self.settings['activation']]
        except KeyError:
            print "ERROR: Activation function \"" + self.settings['activation'] + "\"",
            print "not supported in SKNN."
            raise NotImplementedError

        if self.settings['algorithm'] == 'sgd':
            if self.settings['momentum'] == 0.0:
                learning_rule='sgd'
            elif self.settings['nesterovs_momentum'] == True:
                learning_rule='nesterov'
            else:
                learning_rule='momentum'
        else:
            raise NotImplementedError

        ###############
        #  Create NN  #
        ###############

        nn = sknn_MLPClassifier(
                # NN architecture
                layers=[Layer(activation, units=self.settings['hidden_layer_sizes'][0]),
                        Layer("Softmax")],

                # Learning settings
                learning_rate=self.settings['learning_rate_init'],
                learning_rule=learning_rule,
                learning_momentum=self.settings['momentum'],

                # dropout_rate=0.5,
                # weight_decay=0.0001,
                n_iter=10,
                random_state=self.settings['random_state'],
                verbose=True
        )

        print nn
        nn.fit(X_train, Y_train)
        Y_test_predicted = nn.predict_proba(X_test)
        return Y_test_predicted

    def pybrain(self, X_train, Y_train, X_test):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {'linear': LinearLayer, 'logistic': SigmoidLayer, 'tanh': TanhLayer}
        try:
            hiddenLayerType = activation_dict[self.settings['activation']]
        except KeyError:
            print "ERROR: Activation function \"" + self.settings['activation'] + "\"",
            print "not supported in PyBrain."
            raise NotImplementedError

        if self.settings['nesterovs_momentum'] == True:
            print "ERROR: Nesterov's momentum is not supported in PyBrain."
            raise NotImplementedError

        ###############
        #  Create NN  #
        ###############

        nn = FeedForwardNetwork()

        inLayer = LinearLayer(self.n_features)
        hiddenLayer = hiddenLayerType(self.settings['hidden_layer_sizes'][0])
        outLayer = SoftmaxLayer(self.n_outcomes)

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

        if self.settings['algorithm'] == 'sgd':
            trainer = BackpropTrainer(
                nn,
                learningrate=self.settings['learning_rate_init'],
                dataset=train,
                momentum=self.settings['momentum'],
                verbose=True,
            )
        else:
            print "ERROR: Only SGD is implemented in PyBrain at present."
            raise NotImplementedError

        trainer.trainEpochs(2)

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
