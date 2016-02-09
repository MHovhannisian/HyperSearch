#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Module containing a class which allows access to disparate Python neural
network implementations and architectures, united through a common interface.
This interface is modelled on the scikit-learn interface.
'''

import warnings
import math

from sklearn.cross_validation import KFold, train_test_split
from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier as SKL_MLP

from sknn.mlp import Classifier as sknn_MLPClassifier, Layer

import numpy as np
# NOTE: This is the only simple way to get reproducable results in Keras.
np.random.seed(1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils.visualize_util import plot
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler

class Generic_NN(object):
    ''' Unified interface to compare neural network modules and hyperparameters.

    The module is initialised with arguments that associate it with a dataset.
    Then, neural networks from multiple packages with varying hyperparameters
    can be trained to this dataset and the results compared. Settings for
    neural network architecture and behaviour are taken in a format largely
    matching scikit-learn's MLPClassifier keyword arguments.
    '''

    # Settings which are natively understood by scikit-learn are implemented
    # exactly as in the scikit-learn documentation:
    # http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPClassifier.html
    # Other settings have a comment starting with "# !!"
    _default_nn_settings = {
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
        # self._iter_settings dict.
        'warm_start': True,
        # In scikit-learn (and sknn), "iter" really means epoch.
        'max_iter': 1,
        'learning_rate': 'constant',
    }


    # Iteration settings are homogenised through this dict.
    # This includes stopping conditions and learning rate decay
    # Each module runs for one epoch between the driver having control.
    # Note that these override individual modules' stopping settings.
    _default_iter_settings = {
        # Epochs to run for if no convergence. Note that max iterations
        # are not specified but inferred from max_epoch and batch_size
        'max_epoch': 3,

        # Max decline in loss between epochs to consider converged. (Ratio)
        'epoch_tol': 0.02,

        # Number of consecutive epochs considered converged before
        # stopping.
        'n_stable': 3,

        # For SGD, decay in learning rate between epochs. 0 = no decay.
        'learning_decay': 0.000,

        # Terminate before the loss stops improving if the accuracy score
        # on the validation stops improving. Uses epoch_tol and n_stable.
        'early_stopping': True,
    }

    # For settings which take a categorical value, provided is a dict of
    # which settings should work in which of the Python modules.
    # This dict exists only for reference. It is not used for computaton.
    supported_settings = {
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
        }
    }

    def __init__(self, X, Y, split=(0.75, 0.10, 0.15)):
        ''' Initialisaton tasks:
        - Split and store data as self.{train, validate, test}
        - Set default values for neural networks

        Inputs:
        X - features of dataset
        Y - correct labels of dataset
        split - 3-value tuple of the train/validate/test split

        Returns:
        self
        '''

        # Normalise inputs and split data
        self.X_train, self.X_validate, self.X_test, self.Y_train, self.Y_validate, self.Y_test = \
            self._prepare_data(X, Y, split)

        self.n_features = X.shape[1]
        self.n_outcomes = Y.shape[1]

        # Help Scikit-learn support multi-label classification probabilities
        self.n_labels_sklearn = self.Y_train.sum(axis=1).mean()

        # Stores all tests which have been run.
        self.tests = []

        # Apply the default settings
        self.set_nn_settings(_default_nn_settings)
        self.set_iter_settings(_default_iter_settings)

    @staticmethod
    def _prepare_data(X, Y, split):
        X = np.array(X).astype('float64')
        Y = np.array(Y).astype(bool)

        try:
            assert(sum(split) == 1.0)
        except AssertionError:
            raise AssertionError("Suggested data split doesn't sum to 1.0")

        try:
            assert(X.shape[0] == Y.shape[0])
        except AssertionError:
            raise AssertionError("Number of samples differs between X and Y.")

        split_randint = 0

        X_train, X, Y_train, Y = train_test_split(
            X, Y, test_size=split[1] + split[2], random_state=split_randint)

        X_validate, X_test, Y_validate, Y_test =\
            train_test_split(
                X, Y,
                test_size=split[2] / (split[1] + split[2]),
                random_state=split_randint
            )

        # Train the normaliser on training data only
        normaliser = preprocessing.StandardScaler().fit(X_train)
        X_train = normaliser.transform(X_train)
        X_validate = normaliser.transform(X_validate)
        X_test = normaliser.transform(X_test)

        return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

    def _validate_settings(self):
        ''' Some basic compatibility checks between settings. Doesn't check
        module-specific validity, e.g. whether sklearn supports an algorithm.
        '''

        if self._nn_settings['algorithm'] != 'sgd' and self._iter_settings['learning_decay'] != 0.0:
            raise ValueError(
                "The learning_decay option is for the sgd algorithm only.")

    def get_nn_settings(self):
        return dict(self._nn_settings)

    def get_iter_settings(self):
        return dict(self._iter_settings)

    def set_nn_settings(self, **new_settings):
        ''' Update and re-validate the neural network settings dict '''

        self._nn_settings.update(new_settings)
        self._validate_settings()
        return self

    def set_iter_settings(self, **new_settings):
        ''' Update and re-validate the neural network iteraton settings dict '''

        self._iter_settings.update(new_settings)
        self._validate_settings()
        return self

    def run_test(self, module):
        modules = {
            'sklearn': self._sklearn,
            'sknn': self._sknn,
            'keras': self._keras
        }

        test = {
            'module': module,
            '_nn_settings': self.get_nn_settings(),
            '_iter_settings': self.get_iter_settings()
        }

        F_score = 0.0
        percent_score = 0.0

        score, loss_curve, valid_curve, model = modules[module]()
        test['loss_curve'] = loss_curve
        test['valid_curve'] = valid_curve
        test['model'] = model
        test['score'] = score

        self.tests.append(test)

        return test

    def _keras(self):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {'relu': 'relu', 'linear': 'linear',
                           'logistic': 'sigmoid', 'tanh': 'tanh'}
        try:
            activation = activation_dict[self._nn_settings['activation']]
        except KeyError:
            err_str = "Activation function \"" + self._nn_settings['activation']
            err_str += "\" not supported in Keras."
            raise KeyError(err_str)

        if self._nn_settings['dropout'] != 0.0:
            warnings.warn(
                "I am not convinced that dropout is working correctly in Keras.")

        # Callback for SGD learning rate decline
        n_epoch = [0]

        def learning_schedule(epoch):
            init = self._nn_settings['learning_rate_init']
            factor = (1 - self._iter_settings['learning_decay'])**n_epoch[0]
            lr = factor * init
            return lr

        ###############
        #  Create NN  #
        ###############

        keras_nn = Sequential()

        keras_nn.add(Dense(
            self._nn_settings['hidden_layer_sizes'][0],
            input_dim=self.n_features,
            init='lecun_uniform',
            W_regularizer=l2(self._nn_settings['alpha']),
            activation=activation)
        )

        keras_nn.add(Dropout(self._nn_settings['dropout']))

        keras_nn.add(Dense(
            self.n_outcomes,
            init='lecun_uniform',
            W_regularizer=l2(self._nn_settings['alpha']),
            activation='sigmoid')
        )

        if self._nn_settings['algorithm'] == 'sgd':
            optimiser = SGD(
                lr=self._nn_settings['learning_rate_init'],
                decay=0.0,
                momentum=self._nn_settings['momentum'],
                nesterov=self._nn_settings['nesterovs_momentum'],
            )
        elif self._nn_settings['algorithm'] == 'adam':
            optimiser = Adam(
                lr=self._nn_settings['learning_rate_init'],
                beta_1=self._nn_settings['beta_1'],
                beta_2=self._nn_settings['beta_2'],
                epsilon=self._nn_settings['epsilon']
            )
        elif self._nn_settings['algorithm'] == 'adadelta':
            optimiser = Adadelta()  # Recommended to use the default values
        else:
            err_str = "Learning algorithm \"" + self._nn_settings['algorithm']
            err_str += "\" not implemented in Keras at present."
            raise KeyError(err_str)

        keras_nn.compile(loss='binary_crossentropy', optimizer=optimiser)

        ##############
        #  Train NN  #
        ##############

        loss_curve = []
        valid_curve = []
        n_loss = [0]
        n_valid = [0]
        stop_reason = 0

        for i in range(self._iter_settings['max_epoch']):
            n_epoch[0] = i

            history = keras_nn.fit(
                self.X_train, self.Y_train,
                nb_epoch=10,
                batch_size=self._nn_settings['batch_size'],
                verbose=0,
                callbacks=[LearningRateScheduler(learning_schedule)]
            )

            ####################
            #  Track progress  #
            ####################

            loss_curve.append(history.history['loss'][1])

            valid_proba = keras_nn.predict_proba(self.X_validate, verbose=0)
            valid_score = self.score_from_proba(valid_proba, self.Y_validate)
            valid_curve.append(valid_score)

            #############################
            #  Check stopping criteria  #
            #############################

            if self._converged(loss_curve, n_loss):
                stop_reason = 1
                break

            if self._iter_settings['early_stopping'] and self._converged(valid_curve, n_valid):
                stop_reason = 2
                break

        test_proba = keras_nn.predict_proba(self.X_test, verbose=0)
        score = self.score_from_proba(test_proba, self.Y_test)

        return score, loss_curve, valid_curve, keras_nn

    @staticmethod
    def score_from_proba(proba, Y):
        predictions = (proba > 0.5)
        score = (predictions == Y).mean()
        return score

    def _sklearn(self):

        #####################################################
        #  Strip settings that are unrecognised by sklearn  #
        #####################################################

        unsupported_keys = ['dropout']
        bad_settings = [self._nn_settings[key] > 0 for key in unsupported_keys]

        if any(bad_settings):
            err_str = "The following unsupported settings are not set to 0.0:\n"
            for i, key in enumerate(unsupported_keys):
                if bad_settings[i]:
                    err_str += "\t" + key + ": " + \
                        str(self._nn_settings[key]) + "\n"
            raise KeyError(err_str)

        valid_keys = [
            'hidden_layer_sizes', 'activation', 'alpha', 'batch_size',
            'learning_rate', 'max_iter', 'random_state', 'shuffle', 'tol',
            'learning_rate_init', 'power_t', 'verbose', 'warm_start',
            'momentum', 'nesterovs_momentum', 'validation_fraction',
            'beta_1', 'beta_2', 'epsilon', 'algorithm'
        ]

        sklearn_settings = {key: val for key, val in self._nn_settings.items()
                            if key in valid_keys}

        sklearn_settings.update({'n_labels': self.n_labels_sklearn})

        ###############
        #  Create NN  #
        ###############

        sklearn_nn = SKL_Multilabel_MLP(**sklearn_settings)

        ##############
        #  Train NN  #
        ##############

        # Tracking for stopping criteria and output
        loss_curve = []
        valid_curve = []
        n_loss = [0]
        n_valid = [0]
        stop_reason = 0

        learning_rate = self._nn_settings['learning_rate_init']

        for i in range(self._iter_settings['max_epoch']):
            sklearn_nn.fit(self.X_train, self.Y_train)
            loss_curve = sklearn_nn.loss_curve_  # sklearn itself keeps a list across fits

            learning_rate *= (1.0 - self._iter_settings['learning_decay'])
            sklearn_nn.set_params(learning_rate_init=learning_rate)

            valid_proba = sklearn_nn.predict_proba(self.X_validate)
            valid_score = self.score_from_proba(valid_proba, self.Y_validate)
            valid_curve.append(valid_score)

            #############################
            #  Check stopping criteria  #
            #############################

            if self._converged(loss_curve, n_loss):
                stop_reason = 1
                break

            if self._iter_settings['early_stopping'] and self._converged(valid_curve, n_valid):
                stop_reason = 2
                break

        test_proba = sklearn_nn.predict_proba(self.X_test)
        score = self.score_from_proba(test_proba, self.Y_test)

        return score, loss_curve, valid_curve, sklearn_nn

    def _sknn(self):

        #########################
        #  Settings conversion  #
        #########################

        activation_dict = {
            'relu': 'Rectifier', 'linear': 'Linear', 'logistic': 'Sigmoid', 'tanh': 'Tanh'}
        try:
            activation = activation_dict[self._nn_settings['activation']]
        except KeyError:
            err_str = "Activation function \"" + self._nn_settings['activation']
            err_str += "\" not supported in SKNN."
            raise KeyError(err_str)

        if self._nn_settings['algorithm'] == 'sgd':
            if self._nn_settings['momentum'] == 0.0:
                learning_rule = 'sgd'
            elif self._nn_settings['nesterovs_momentum'] is True:
                learning_rule = 'nesterov'
            else:
                learning_rule = 'momentum'
        elif self._nn_settings['algorithm'] == 'adadelta':
            learning_rule = 'adadelta'
        else:
            raise KeyError(
                "Only SGD and Adadelta implemented in Scikit-NN at present.")

        if self._iter_settings['learning_decay'] != 0.0:
            raise KeyError(
                "SGD learning decay not supported in SKNN (!)")

        if self._nn_settings['alpha'] != 0.0:
            warnings.warn(
                "I am not convinced that L2 is working correctly in SKNN.")

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
            layers=[Layer(activation, units=self._nn_settings['hidden_layer_sizes'][0],),
                    Layer("Softmax", units=2 * self.n_outcomes)],

            # Learning settings
            loss_type='mcc',
            learning_rate=self._nn_settings['learning_rate_init'],
            learning_rule=learning_rule,
            learning_momentum=self._nn_settings['momentum'],
            batch_size=self._nn_settings['batch_size'],
            n_iter=1,

            # Regularisation
            weight_decay=self._nn_settings['alpha'],
            dropout_rate=self._nn_settings['dropout'],

            random_state=self._nn_settings['random_state'],

            # Callback to get loss
            callback={'on_batch_finish': batch_callback},

            # verbose=1
        )

        print sknn_nn

        ##############
        #  Train NN  #
        ##############

        loss_curve = []
        valid_curve = []
        n_loss = [0]
        n_valid = [0]
        stop_reason = 0

        for i in range(self._iter_settings['max_epoch']):
            sknn_nn.fit(self.X_train, self.Y_train)
            loss_curve.append(batch_loss[0])

            # NOTE: predict_proba returns 2 entries per binary class, which are
            # True and False, adding to 1.0. We take the probability of True.
            valid_proba = sknn_nn.predict_proba(self.X_validate)[:, 1::2]
            valid_score = self.score_from_proba(valid_proba, self.Y_validate)
            valid_curve.append(valid_score)

            # Use change in loss_curve to evaluate stability
            if self._converged(loss_curve, n_loss):
                stop_reason = 1
                break

            if self._iter_settings['early_stopping'] and self._converged(valid_curve, n_valid):
                stop_reason = 2
                break

        predict_proba = sknn_nn.predict_proba(self.X_test)[:, 1::2]
        score = self.score_from_proba(predict_proba, self.Y_test)

        return score, loss_curve, valid_curve, sknn_nn

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

    def _converged(self, objective, n_objective):
        ''' Inputs:

            objective: Loss or validation score at end of each epoch until now
            n_objective: Length of previous stability streak

            Returns:

            True if converged according to settings
            False if not converged according to settings

            Updates n_objective as a side-effect.
        '''

        try:
            objective_ratio = math.fabs(1 - (objective[-1] / objective[-2]))
        except IndexError:
            return False

        if objective_ratio < self._iter_settings['epoch_tol']:
            n_objective[0] += 1
        else:
            n_objective[0] = 0

        if n_objective[0] == self._iter_settings['n_stable']:
            return True
        else:
            return False

class SKL_Multilabel_MLP(SKL_MLP):
    ''' Wrapper for Scikit-learn enabling multi-label probability output. '''

    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 algorithm='adam', alpha=0.0001,
                 batch_size=200, learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_labels=1):

        self.n_labels = n_labels

        sup = super(SKL_Multilabel_MLP, self)
        sup.__init__(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation, algorithm=algorithm, alpha=alpha,
                     batch_size=batch_size, learning_rate=learning_rate,
                     learning_rate_init=learning_rate_init, power_t=power_t,
                     max_iter=max_iter, shuffle=shuffle,
                     random_state=random_state, tol=tol, verbose=verbose,
                     warm_start=warm_start, momentum=momentum,
                     nesterovs_momentum=nesterovs_momentum,
                     early_stopping=early_stopping,
                     validation_fraction=validation_fraction,
                     beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : array-like, shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """

        proba = super(SKL_Multilabel_MLP, self).predict_proba(X)
        return proba*self.n_labels
