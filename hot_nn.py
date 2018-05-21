"""Author: WKirgsn, 2017"""
import sqlite3
import uuid
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, SimpleRNN
from keras.layers.core import Dense, Dropout, Flatten
import keras.optimizers as opts
from keras import regularizers
from keras import __version__ as keras_version
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, LearningRateScheduler

import preprocessing.config as cfg
from preprocessing.data import DataManager, ReSamplerForBatchTraining
import preprocessing.file_utils as futils


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


def build_keras_model(x_shape=(100, 1, 10),
                      arch='lstm',
                      n_layers=1,
                      n_units=64,
                      kernel_reg=None,
                      activity_reg=None,
                      recurrent_reg=None,
                      dropout_rate=0.5,
                      optimizer='nadam',
                      lr_rate=1e-5,
                      ):
    arch_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    arch_dict_cudnn = {'lstm': CuDNNLSTM, 'gru': CuDNNGRU, 'rnn': SimpleRNN}

    opts_map = {'adam': opts.Adam, 'nadam': opts.Nadam,
                'adamax': opts.Adamax, 'sgd': opts.SGD,
                'rmsprop': opts.RMSprop}

    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # before Keras / Tensorflow is imported.

    ann_cfg = {
        'units': n_units,
        'batch_input_shape': (batch_size, x_shape[1], x_shape[2]),
        'kernel_regularizer': kernel_reg,
        'activity_regularizer': activity_reg,
        'recurrent_regularizer': recurrent_reg,
        'stateful': True,
    }
    if n_layers > 1:
        ann_cfg['return_sequences'] = True

    if gpu_available:
        ANN = arch_dict_cudnn[arch]
    else:
        ANN = arch_dict[arch]
        ann_cfg['implementation'] = 2

    # create model
    model = Sequential()
    model.add(ANN(**ann_cfg))
    model.add(Dropout(0.5))
    if n_layers > 1:
        for i in range(n_layers-1):
            ann_cfg.pop('batch_input_shape', None)
            if i == n_layers-2:
                ann_cfg['return_sequences'] = False
            model.add(ANN(**ann_cfg))
            model.add(Dropout(dropout_rate))

    model.add(Dense(4))

    opt = opts_map[optimizer](lr=lr_rate)
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
    batch_size = cfg.keras_cfg['params']['batch_size']
    n_epochs = cfg.keras_cfg['params']['n_epochs']
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

    # todo: How to adaptively decrease lr? -> scheduler callback
    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=cfg.keras_cfg['params']['early_stop_patience'],
                      verbose=0),
    ]

    KerasRegressor_config = {'x_shape': (batch_size, 1, len(dm.cl.x_cols)),
                             'epochs': n_epochs,
                             'batch_size': batch_size,
                             'n_layers': 3,
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
        try:
            plt.subplot(211)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.subplot(212)
            plot_results(actual, inversed_pred)
        except Exception:
            print("Plotting failed..")