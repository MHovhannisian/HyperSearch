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
from keras.optimizers import SGD, Adam
from keras.utils.visualize_util import plot
from keras.regularizers import l2

np.random.seed(1)
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

        # Settings which are natively understood by scikit-learn are implemented
        # exactly as in the scikit-learn documentation:
        # http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPClassifier.html
        # Other settings have a comment starting with "# !!"
        self.settings = {
            ##################
            #  Architecture  #
            ##################
            'hidden_layer_sizes' : (15,),
            'activation' : 'relu',

            ####################
            #  Regularisation  #
            ####################
            'alpha' : 0.0000, # L2 penalty. 0.0 = turned off.
            'dropout' : 0.0, # !! Dropout between hidden and output layers.

            ##############
            #  Learning  #
            ##############
            'learning_rate_init' : 0.001,
            'algorithm' : 'sgd',
            'batch_size' : 16,

            # SGD only
            'momentum' : 0.9,
            'nesterovs_momentum' : False,
            'learning_rate' : 'constant',
            # !! For learning_rate='factor' in Keras/PyBrain, implemented so
            # very low is like 'constant'
            'learning_decay' : 0.000,
            # Only for Scikit-learn's 'learning_rate':'invscaling'
            'power_t' : 0.5,

            # Adam only (Scikit-learn and Keras only)
            'beta_1' : 0.9,
            'beta_2' : 0.999,
            'epsilon' : 1e-8,

            #######################
            #  Consistent output  # (for developing and debugging)
            #######################
            # Doesn't work in Keras (there's a hack though -- see the imports)
            # Doesn't work in PyBrain and no known hack :(
            'random_state' : 1,

            #######################################
            #  Stopping criteria - DO NOT CHANGE  #
            #######################################
            # Stopping is handled manually by this class through the
            # self.stopping_settings dict. This section is just to set up
            # scikt-learn correctly. They are placed here for visiblity.
            'warm_start' : True,
            'max_iter' : 1, # In scikit-learn (and sknn), "iter" really means epoch.
        }

        # Stopping conditions are homogenised through this dict.
        # Note that these differ from sklearn's stopping criteria.
        self.stopping_settings = {
            # Epochs to run for if no convergence. Note that max iterations
            # are not specified but inferred from max_epoch and batch_size
            'max_epoch' : 3,

            # Max decline in loss between epochs to consider converged. (Ratio)
            'epoch_tol' : 0.02,

            # Number of consecutive epochs considered converged before stopping.
            'n_stable' : 3,
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
            },

            'nesterovs_momentum' : {
                True : ['sklearn', 'sknn', 'keras'],
                False : ['sklearn', 'sknn', 'pybrain', 'keras']
            },

            'algorithm' : {
                'sgd' : ['sklearn', 'sknn', 'pybrain', 'keras'],
                'adam' : ['sklearn', 'keras']
            },

            'learning_rate' : { # How the learning rate changes. Only sgd.
                'constant' : ['sklearn', 'sknn', 'pybrain', 'keras'],
                'invscaling' : ['sklearn'],
                'adapative' : ['sklearn'],
                'factor' : ['keras', 'pybrain']
            }
        }

        self.k_folds = k_folds
        self.k_fold = KFold(self.n_samples, n_folds=k_folds, shuffle=True,
                            random_state=self.kfold_seed)

    def get_settings(self):
        return self.settings

    def set_settings(self, new_settings):
        ''' Set the main neural network dict's settings '''
        self.settings.update(new_settings)
        return self

    def set_stopping(self, new_settings):
        ''' Set the stopping criterion dict's settings '''
        self.stopping_settings.update(new_settings)
        return self

    def run_test(self, nntype):
        nntypes = {
            'sklearn' : self.sklearn,
            'sknn'    : self.sknn,
            'pybrain' : self.pybrain,
            'keras'   : self.keras
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

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {'relu': 'relu', 'linear': 'linear', 'logistic': 'sigmoid', 'tanh': 'tanh'}
        try:
            activation = activation_dict[self.settings['activation']]
        except KeyError:
            print "ERROR: Activation function \"" + self.settings['activation'] + "\"",
            print "not supported in Keras."
            raise NotImplementedError

        if self.settings['learning_rate'] == 'factor':
            sgd_decay = self.settings['learning_decay']
        elif self.settings['learning_rate'] == 'constant':
            sgd_decay = 0.0
        else:
            print "ERROR: Learning decay style \"" + self.settings['learning_rate'] + "\"",
            print "not supported in Keras."
            raise NotImplementedError

        if self.settings['dropout'] != 0.0:
            print "WARNING: I am not convinced that dropout is working correctly in Keras."

        ###############
        #  Create NN  #
        ###############

        keras_nn = Sequential()

        keras_nn.add(Dense(
            self.settings['hidden_layer_sizes'][0],
            input_dim=self.n_features,
            init='lecun_uniform',
            W_regularizer=l2(self.settings['alpha']),
            activation=activation)
        )

        keras_nn.add(Dropout(self.settings['dropout']))

        keras_nn.add(Dense(
            self.n_outcomes,
            init='lecun_uniform',
            W_regularizer=l2(self.settings['alpha']),
            activation='softmax')
        )

        if self.settings['algorithm'] == 'sgd':
            optimiser = SGD(
                lr=self.settings['learning_rate_init'],
                decay=sgd_decay,
                momentum=self.settings['momentum'],
                nesterov=self.settings['nesterovs_momentum'],
            )
        elif self.settings['algorithm'] == 'adam':
            optimiser = Adam(
                lr=self.settings['learning_rate_init'],
                beta_1=self.settings['beta_1'],
                beta_2=self.settings['beta_2'],
                epsilon=self.settings['epsilon']
            )
        else:
            print "ERROR:", self.settings['algorithm'], "not implemented in Keras at present."
            raise NotImplementedError

        keras_nn.compile(loss='categorical_crossentropy', optimizer=optimiser)

        ##############
        #  Train NN  #
        ##############

        loss = []
        n_loss = [0]
        stop_reason = 0

        for i in range(self.stopping_settings['max_epoch']):
            history = keras_nn.fit(
                X_train, Y_train,
                nb_epoch=1,
                batch_size=self.settings['batch_size'],
                verbose=0,
            )

            loss.append(history.history['loss'][0])

            # Use change in loss to evaluate stability
            if self.converged(loss, n_loss):
                stop_reason = 1
                break

        print loss
        print len(loss)
        keras_predictions = keras_nn.predict_proba(X_test, verbose=0)

        return keras_predictions

    def sklearn(self, X_train, Y_train, X_test):

        #####################################################
        #  Strip settings that are unrecognised by sklearn  #
        #####################################################

        unsupported_keys = ['dropout', 'learning_decay']
        bad_settings = [self.settings[key] > 0 for key in unsupported_keys]

        if any(bad_settings):
            print "ERROR: The following settings which are unsupported by",
            print "scikit-learn are currently not set to 0.0:"
            for i, key in enumerate(unsupported_keys):
                if bad_settings[i]:
                    print key + ":", self.settings[key]
            raise NotImplementedError

        valid_keys = [
            'hidden_layer_sizes', 'activation', 'alpha', 'batch_size',
            'learning_rate', 'max_iter', 'random_state', 'shuffle', 'tol',
            'learning_rate_init', 'power_t', 'verbose', 'warm_start',
            'momentum', 'nesterovs_momentum', 'early_stopping',
            'validation_fraction', 'beta_1', 'beta_2', 'epsilon', 'algorithm'
        ]

        sklearn_settings = {key: val for key, val in self.settings.items() if key in valid_keys}

        ###############
        #  Create NN  #
        ###############

        sklearn_nn = sklearn_MLPClassifier(**sklearn_settings)
        print sklearn_nn

        ##############
        #  Train NN  #
        ##############

        loss = []
        n_loss = [0]
        stop_reason = 0

        for i in range(self.stopping_settings['max_epoch']):
            sklearn_nn.fit(X_train, Y_train)
            loss = sklearn_nn.loss_curve_ # sklearn itself keeps a list across fits

            # Use change in loss to evaluate stability
            if self.converged(loss, n_loss):
                stop_reason = 1
                break

        print loss
        print len(loss)
        Y_test_predicted = sklearn_nn.predict_proba(X_test)
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
            print "ERROR: Only SGD is implemented in Scikit-NN at present."
            raise NotImplementedError

        if self.settings['learning_rate'] != 'constant':
            print "ERROR: Learning decay not supported in SKNN (!)"
            raise NotImplementedError

        if self.settings['alpha'] != 0.0:
            print "WARNING: I am not convinced that L2 is working in SKNN"
            print "Dropout works, though."

        # The contents of a mutable variable can be changed in a closure.
        # SKNN doesn't give access to the loss in the end-of-epoch callback,
        # only in the end-of-batch callback.
        batch_loss = [0]
        def batch_callback(**variables):
            batch_loss[0] = variables['loss']/variables['count']

        ###############
        #  Create NN  #
        ###############

        sknn_nn = sknn_MLPClassifier(

            # sknn_nn architecture
            layers=[Layer(activation, units=self.settings['hidden_layer_sizes'][0],),
                    Layer("Softmax", units=2*self.n_outcomes)],

            # Learning settings
            loss_type='mcc',
            learning_rate=self.settings['learning_rate_init'],
            learning_rule=learning_rule,
            learning_momentum=self.settings['momentum'],
            batch_size=self.settings['batch_size'],
            n_iter=1,

            # Regularisation
            weight_decay=self.settings['alpha'],
            dropout_rate=self.settings['dropout'],

            random_state=self.settings['random_state'],

            # Callback to get loss
            callback = {'on_batch_finish': batch_callback},

            verbose=1
        )

        print sknn_nn

        ##############
        #  Train NN  #
        ##############

        loss = []
        n_loss = [0]
        stop_reason = 0

        for i in range(self.stopping_settings['max_epoch']):
            sknn_nn.fit(X_train, Y_train)
            loss.append(batch_loss[0])

            # Use change in loss to evaluate stability
            if self.converged(loss, n_loss):
                stop_reason = 1
                break

        print X_train.shape
        print loss
        print len(loss)

        # NOTE: predict_proba returns 2 entries per binary class, which are
        # True and False and add to give 1.0. We take the probability of True.
        Y_test_predicted = sknn_nn.predict_proba(X_test)[:, 1::2]

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

        if self.settings['algorithm'] != 'sgd':
            print "ERROR: Only weight optimisation algorithm 'sgd' is",
            print "supported in PyBrain."
            raise NotImplementedError

        if self.settings['learning_rate'] == 'factor':
            # Unlike in Keras, 1.0 is off and 0.0 is maximum learning decay
            # To keep 0.0 as off, we take 1.0 - learning_decay
            sgd_decay = 1.0 - self.settings['learning_decay']
        elif self.settings['learning_rate'] == 'constant':
            sgd_decay = 1.0
        else:
            print "ERROR: Learning decay style \"" + self.settings['learning_rate'] + "\"",
            print "not supported in PyBrain."
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
                weightdecay=self.settings['alpha'],
                lrdecay=sgd_decay,
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
        # print predicted_positives
        # print answers
        true_positives = (predicted_positives & answers)
        false_positives = (predicted_positives & np.logical_not(answers))
        correct_predictions = (predictions == answers)

        precision = float(true_positives.sum())/predicted_positives.sum()
        recall = float(true_positives.sum())/answers.sum()
        F1 = (2*precision*recall)/(precision+recall)
        score = float(correct_predictions.sum())/(predictions.shape[0]*predictions.shape[1])

        return F1, score

    def converged(self, loss, n_loss):
        ''' Inputs:

            loss: Loss at end of each epoch until now
            n_loss: Length of previous stability streak

            Returns:

            True if converged according to settings
            False if not converged according to settings

            Updates n_loss as a side-effect.
        '''

        try:
            loss_ratio = 1 - (loss[-1]/loss[-2])
        except IndexError:
            return False

        if loss_ratio < self.stopping_settings['epoch_tol']:
            n_loss[0] += 1
        else:
            n_loss[0] = 0

        if n_loss[0] == self.stopping_settings['n_stable']:
            return True
        else:
            return False

