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

import numpy as np
# NOTE: This is the only simple way to get reproducable results in Keras.
np.random.seed(1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adadelta
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

        # Stores all tests which have been run, will contain a dict attributes
        self.tests = []

        # Settings which are natively understood by scikit-learn are implemented
        # exactly as in the scikit-learn documentation:
        # http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPClassifier.html
        # Other settings have a comment starting with "# !!"
        self.nn_settings = {
            ##################
            #  Architecture  #
            ##################
            'hidden_layer_sizes': (15,),
            'activation': 'relu',

            ####################
            #  Regularisation  #
            ####################
            'alpha': 0.0000,  # L2 penalty. 0.0 = turned off.
            'dropout': 0.0,  # !! Dropout between hidden and output layers.

            ##############
            #  Learning  #
            ##############
            'learning_rate_init': 0.001,
            'algorithm': 'sgd',
            'batch_size': 16,

            # SGD only
            'momentum': 0.9,
            'nesterovs_momentum': False,

            # Adam only (Scikit-learn and Keras only)
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,

            #######################
            #  Consistent output  # (for developing and debugging)
            #######################
            # Doesn't work in Keras (there's a hack though -- see the imports)
            'random_state': 1,

            ###############################################
            #  Iterations/epoch settings - DO NOT CHANGE  #
            ###############################################
            # Training iteration is handled manually by this class through the
            # self.iter_settings dict.
            'warm_start': True,
            # In scikit-learn (and sknn), "iter" really means epoch.
            'max_iter': 1,
            'learning_rate': 'constant',
        }

        # Iteration settings are homogenised through this dict.
        # This includes stopping conditions and learning rate decay
        # Each module runs for one epoch between the driver having control.
        # Note that these override individual modules' stopping settings.
        self.iter_settings = {
            # Epochs to run for if no convergence. Note that max iterations
            # are not specified but inferred from max_epoch and batch_size
            'max_epoch': 3,

            # Max decline in loss between epochs to consider converged. (Ratio)
            'epoch_tol': 0.02,

            # Number of consecutive epochs considered converged before
            # stopping.
            'n_stable': 3,

            # For SGD, decay in learning rate between epochs. 0 = no decay.
            # TODO NOTIMPLEMENTEDYET
            # only implemented in sklearn
            'learning_decay': 0.000,
        }

        # For settings which take a categorical value, provided is a dict of
        # which settings should work in which of the Python modules.
        # This dict exists only for reference. It is not used for computaton.
        self.supported_settings = {
            'activation': {
                'relu': ['sklearn', 'sknn', 'keras'],
                'linear': ['sknn', 'keras'],
                'logistic': ['sklearn', 'sknn', 'keras'],
                'tanh': ['sklearn', 'sknn', 'keras']
            },

            'nesterovs_momentum': {
                True: ['sklearn', 'sknn', 'keras'],
                False: ['sklearn', 'sknn', 'keras']
            },

            'algorithm': {
                'sgd': ['sklearn', 'sknn', 'keras'],
                'adam': ['sklearn', 'keras'],
                'adadelta': ['sknn', 'keras']
            },
        }

        self._validate_settings()

        self.k_folds = k_folds
        self.k_fold = KFold(self.n_samples, n_folds=k_folds, shuffle=True,
                            random_state=self.kfold_seed)

    def _validate_settings(self):
        ''' Some basic compatibility checks between settings. Doesn't check
        module-specific validity, e.g. whether sklearn supports an algorithm.
        '''

        if self.nn_settings['algorithm'] != 'sgd' and self.iter_settings['learning_decay'] != 0.0:
            raise ValueError(
                "The learning_decay option is for the sgd algorithm only.")

    def get_nn_settings(self):
        return self.nn_settings

    def get_iter_settings(self):
        return self.iter_settings

    def set_nn_settings(self, new_settings):
        ''' Update and re-validate the neural network settings dict '''

        self.nn_settings.update(new_settings)
        self._validate_settings()
        return self

    def set_iter_settings(self, new_settings):
        ''' Set the iteration settings dict's settings '''

        self.iter_settings.update(new_settings)
        self._validate_settings()
        return self

    def run_test(self, module):
        modules = {
            'sklearn': self.sklearn,
            'sknn': self.sknn,
            'keras': self.keras
        }

        test = {'module': module,
                'nn_settings': dict(self.nn_settings),
                'iter_settings': dict(self.iter_settings)
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

            predict_proba, loss_curve, valid_curve, model = modules[
                module](X_train, Y_train, X_test)
            test['predict_proba'] = predict_proba
            test['loss_curve'] = loss_curve
            test['valid_curve'] = valid_curve
            test['model'] = model

            # score = self.get_score(Y_test_predicted, Y_test)
            # F_score += score[0]/self.k_folds
            # percent_score += score[1]/self.k_folds
            break

        self.tests.append(test)

        # return F_score, percent_score
        # return Y_test_predict, Y_test_predict_proba, Y_test, model, X_test
        return test['predict_proba']

    def keras(self, X_train, Y_train, X_test):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {'relu': 'relu', 'linear': 'linear',
                           'logistic': 'sigmoid', 'tanh': 'tanh'}
        try:
            activation = activation_dict[self.nn_settings['activation']]
        except KeyError:
            err_str = "Activation function \"" + self.nn_settings['activation']
            err_str += "\" not supported in Keras."
            raise NotImplementedError(err_str)

        if self.nn_settings['dropout'] != 0.0:
            print "WARNING: I am not convinced that dropout is working correctly in Keras."

        ###############
        #  Create NN  #
        ###############

        keras_nn = Sequential()

        keras_nn.add(Dense(
            self.nn_settings['hidden_layer_sizes'][0],
            input_dim=self.n_features,
            init='lecun_uniform',
            W_regularizer=l2(self.nn_settings['alpha']),
            activation=activation)
        )

        keras_nn.add(Dropout(self.nn_settings['dropout']))

        keras_nn.add(Dense(
            self.n_outcomes,
            init='lecun_uniform',
            W_regularizer=l2(self.nn_settings['alpha']),
            activation='softmax')
        )

        if self.nn_settings['algorithm'] == 'sgd':
            optimiser = SGD(
                lr=self.nn_settings['learning_rate_init'],
                decay=0.0,
                momentum=self.nn_settings['momentum'],
                nesterov=self.nn_settings['nesterovs_momentum'],
            )
        elif self.nn_settings['algorithm'] == 'adam':
            optimiser = Adam(
                lr=self.nn_settings['learning_rate_init'],
                beta_1=self.nn_settings['beta_1'],
                beta_2=self.nn_settings['beta_2'],
                epsilon=self.nn_settings['epsilon']
            )
        else:
            err_str = "Learning algorithm \"" + self.nn_settings['algorithm']
            err_str += "\" not implemented in Keras at present."
            raise NotImplementedError(err_str)

        keras_nn.compile(loss='categorical_crossentropy',
                         class_mode='binary', optimizer=optimiser)

        ##############
        #  Train NN  #
        ##############

        loss_curve = []
        valid_curve = []
        n_loss = [0]
        stop_reason = 0

        for i in range(self.iter_settings['max_epoch']):
            history = keras_nn.fit(
                X_train, Y_train,
                nb_epoch=1,
                batch_size=self.nn_settings['batch_size'],
                verbose=0,
            )

            loss_curve.append(history.history['loss'][0])

            # Use change in loss_curve to evaluate stability
            if self.converged(loss_curve, n_loss):
                stop_reason = 1
                break

        keras_predict_proba = keras_nn.predict_proba(X_test, verbose=0)

        return keras_predict_proba, loss_curve, valid_curve, keras_nn

    def sklearn(self, X_train, Y_train, X_test):

        #####################################################
        #  Strip settings that are unrecognised by sklearn  #
        #####################################################

        unsupported_keys = ['dropout']
        bad_settings = [self.nn_settings[key] > 0 for key in unsupported_keys]

        if any(bad_settings):
            err_str = "The following unsupported settings are not set to 0.0:\n"
            for i, key in enumerate(unsupported_keys):
                if bad_settings[i]:
                    err_str += "\t" + key + ": " + \
                        str(self.nn_settings[key]) + "\n"
            raise NotImplementedError(err_str)

        valid_keys = [
            'hidden_layer_sizes', 'activation', 'alpha', 'batch_size',
            'learning_rate', 'max_iter', 'random_state', 'shuffle', 'tol',
            'learning_rate_init', 'power_t', 'verbose', 'warm_start',
            'momentum', 'nesterovs_momentum', 'early_stopping',
            'validation_fraction', 'beta_1', 'beta_2', 'epsilon', 'algorithm'
        ]

        sklearn_settings = {key: val for key, val in self.nn_settings.items()
                            if key in valid_keys}

        ###############
        #  Create NN  #
        ###############

        sklearn_nn = sklearn_MLPClassifier(**sklearn_settings)
        print sklearn_nn

        ##############
        #  Train NN  #
        ##############

        # Tracking for stopping criteria and output
        loss_curve = []
        n_loss = [0]
        stop_reason = 0
        valid_curve = []

        learning_rate = self.nn_settings['learning_rate_init']

        for i in range(self.iter_settings['max_epoch']):
            sklearn_nn.fit(X_train, Y_train)
            loss_curve = sklearn_nn.loss_curve_  # sklearn itself keeps a list across fits

            learning_rate *= (1.0 - self.iter_settings['learning_decay'])
            sklearn_nn.set_params(learning_rate_init=learning_rate)

            # Use change in loss_curve to evaluate stability
            if self.converged(loss_curve, n_loss):
                stop_reason = 1
                break

        print loss_curve
        print len(loss_curve)
        predict_proba = sklearn_nn.predict_proba(X_test)

        return predict_proba, loss_curve, valid_curve, sklearn_nn

    def sknn(self, X_train, Y_train, X_test):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {
            'relu': 'Rectifier', 'linear': 'Linear', 'logistic': 'Sigmoid', 'tanh': 'Tanh'}
        try:
            activation = activation_dict[self.nn_settings['activation']]
        except KeyError:
            err_str = "Activation function \"" + self.nn_settings['activation']
            err_str += "\" not supported in SKNN."
            raise NotImplementedError(err_str)

        if self.nn_settings['algorithm'] == 'sgd':
            if self.nn_settings['momentum'] == 0.0:
                learning_rule = 'sgd'
            elif self.nn_settings['nesterovs_momentum'] == True:
                learning_rule = 'nesterov'
            else:
                learning_rule = 'momentum'
        else:
            raise NotImplementedError(
                "Only SGD is implemented in Scikit-NN at present.")

        if self.nn_settings['alpha'] != 0.0:
            print "WARNING: I am not convinced that L2 is working in SKNN"
            print "Dropout works, though."

        # The contents of a mutable variable can be changed in a closure.
        # SKNN doesn't give access to the loss in the end-of-epoch callback,
        # only in the end-of-batch callback.
        batch_loss = [0]

        def batch_callback(**variables):
            batch_loss[0] = variables['loss'] / variables['count']

        ###############
        #  Create NN  #
        ###############

        sknn_nn = sknn_MLPClassifier(

            # sknn_nn architecture
            layers=[Layer(activation, units=self.nn_settings['hidden_layer_sizes'][0],),
                    Layer("Softmax", units=2 * self.n_outcomes)],

            # Learning settings
            loss_type='mcc',
            learning_rate=self.nn_settings['learning_rate_init'],
            learning_rule=learning_rule,
            learning_momentum=self.nn_settings['momentum'],
            batch_size=self.nn_settings['batch_size'],
            n_iter=1,

            # Regularisation
            weight_decay=self.nn_settings['alpha'],
            dropout_rate=self.nn_settings['dropout'],

            random_state=self.nn_settings['random_state'],

            # Callback to get loss
            callback={'on_batch_finish': batch_callback},

            # verbose=1
        )

        print sknn_nn

        ##############
        #  Train NN  #
        ##############

        loss_curve = []
        n_loss = [0]
        stop_reason = 0
        valid_curve = []

        for i in range(self.iter_settings['max_epoch']):
            sknn_nn.fit(X_train, Y_train)
            loss_curve.append(batch_loss[0])

            # Use change in loss_curve to evaluate stability
            if self.converged(loss_curve, n_loss):
                stop_reason = 1
                break

        print X_train.shape
        print loss_curve
        print len(loss_curve)

        # NOTE: predict_proba returns 2 entries per binary class, which are
        # True and False and add to give 1.0. We take the probability of True.
        predict_proba = sknn_nn.predict_proba(X_test)[:, 1::2]

        return predict_proba, loss_curve, valid_curve, sknn_nn

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

        precision = float(true_positives.sum()) / predicted_positives.sum()
        recall = float(true_positives.sum()) / answers.sum()
        F1 = (2 * precision * recall) / (precision + recall)
        score = float(correct_predictions.sum()) / \
            (predictions.shape[0] * predictions.shape[1])

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
            loss_ratio = 1 - (loss[-1] / loss[-2])
        except IndexError:
            return False

        if loss_ratio < self.iter_settings['epoch_tol']:
            n_loss[0] += 1
        else:
            n_loss[0] = 0

        if n_loss[0] == self.iter_settings['n_stable']:
            return True
        else:
            return False
