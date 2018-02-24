"""Author: WKirgsn, 2017"""
import sqlite3
import uuid
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, SimpleRNN
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam, Nadam
from keras import regularizers
from keras import __version__ as keras_version
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

import kirgsn.config as cfg
from kirgsn.data import DataManager, ReSamplerForBatchTraining
import kirgsn.file_utils as futils


class CustomKerasRegressor(KerasRegressor):
    def reset_states(self):
        self.model.reset_states()


def reshape_input_for_batch_train(x_set):
    assert isinstance(x_set, pd.DataFrame)
    x = x_set.as_matrix()
    return np.reshape(x, (x.shape[0], 1, x.shape[1]))


def plot_results(_y, _yhat):
    plt.plot(_y, alpha=0.6, color='darkorange')
    plt.plot(_yhat, lw=2, color='navy')
    plt.show()


def build_keras_model(x_shape=(100, 1, 10)):
    print('Keras version: {}'.format(keras_version))
    n_neurons = cfg.keras_cfg['n_neurons']
    arch_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    arch_dict_cudnn = {'lstm': CuDNNLSTM, 'gru': CuDNNGRU, 'rnn': SimpleRNN}
    if gpu_available:
        ANN = arch_dict_cudnn[cfg.keras_cfg['arch']]
    else:
        ANN = arch_dict[cfg.keras_cfg['arch']]

    # create model
    model = Sequential()
    model.add(
        ANN(n_neurons,
            # implementation=2,  # only known by non-CUDNN classes
            batch_input_shape=(batch_size, x_shape[1], x_shape[2]),
            kernel_regularizer=regularizers.l2(cfg.train_cfg['l2_reg_w']),
            activity_regularizer=regularizers.l2(cfg.train_cfg['l2_reg_w']),
            recurrent_regularizer=regularizers.l2(
                cfg.train_cfg['l2_reg_w']),
            stateful=True,
            ))
    model.add(Dropout(0.5))
    model.add(Dense(4))

    opt = Nadam()
    model.compile(optimizer=opt, loss='mse')
    return model


if __name__ == '__main__':

    # config
    gpu_available = len(futils.get_available_gpus()) >= 1
    if cfg.debug_cfg['choose_debug_on_gpu_availability']:
        DEBUG = not gpu_available
    else:
        DEBUG = cfg.debug_cfg['DEBUG']
    if DEBUG:
        print('## DEBUG MODE ON ##')
    n_debug = cfg.debug_cfg['n_debug']
    batch_size = cfg.keras_cfg['batch_size']
    n_epochs = cfg.keras_cfg['n_epochs']
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager(cfg.data_cfg['file_path'])

    # featurize dataset (feature engineering)
    tra_df, val_df, tst_df = dm.get_featurized_sets()

    # reorder samples
    resampler = ReSamplerForBatchTraining(batch_size)
    tra_df = resampler.fit_transform(tra_df)  # there is nothing fitted
    val_df = resampler.transform(val_df)
    tst_df = resampler.transform(tst_df)

    # todo: How to adaptively decrease lr?
    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=cfg.train_cfg['early_stop_patience'],
                      verbose=0),
    ]

    KerasRegressor_config = {'x_shape': (batch_size, 1, len(dm.cl.x_cols)),
                             'epochs': n_epochs,
                             'batch_size': batch_size,
                             'validation_data': (
                                 reshape_input_for_batch_train(
                                     val_df[dm.cl.x_cols]),
                                 val_df[dm.cl.y_cols]),
                             'verbose': 1,
                             'shuffle': False,
                             'callbacks': callbacks,
                             }

    nn_estimator = CustomKerasRegressor(build_fn=build_keras_model,
                                        **KerasRegressor_config)

    # fit
    history = nn_estimator.fit(
        reshape_input_for_batch_train(tra_df[dm.cl.x_cols]),
        tra_df[dm.cl.y_cols])
    nn_estimator.reset_states()

    # predict
    yhat = nn_estimator.predict(
        reshape_input_for_batch_train(tst_df[dm.cl.x_cols]),
        batch_size=batch_size)

    # correct the order again
    yhat = resampler.inverse_transform(yhat)

    # compare with actual
    actual = dm.actual
    inversed_pred = dm.inverse_prediction(yhat)
    # trunc actual if prediction is shorter
    if yhat.shape[0] != actual.shape[0]:
        print('trunc actual from {} to {} samples'.format(actual.shape[0],
                                                          yhat.shape[0]))
        offset = actual.shape[0] - yhat.shape[0]
        actual = actual.iloc[offset:, :]
    print('mse: {:.6} K²'.format(mean_squared_error(actual.values,
                                                    inversed_pred.values)))
    # save predictions
    futils.save_predictions(model_uuid, inversed_pred)
    # plots
    if cfg.plot_cfg['do_plot']:
        plt.subplot(211)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.subplot(212)
        plot_results(actual, inversed_pred)