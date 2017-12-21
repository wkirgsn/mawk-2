"""Author: WKirgsn, 2017"""
from os.path import join
import warnings
import multiprocessing
import time
import os

from tensorflow.python.client import device_lib
import numpy as np

import config as cfg

warnings.filterwarnings("ignore")
np.random.seed(2017)

# todo: taskset only if not 0xfff
# todo: refactor to stateful RNNs
# todo: add shifts to axis 0 (figure out how many shifts)
# todo: feature engineer similar to two sigma financial competition winners
# todo: add dropout/ l1, l2 regularization
# todo: save train trend history for visualizing
# todo: apply bayesian opt for hyperparams


def measure_time(func):
    """time measuring decorator"""
    def wrapped(*args, **kwargs):

        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('took {} seconds'.format(end_time-start_time))
        return ret
    return wrapped


def _prll_create_dataset(idx, obs, data, x_cols, y_cols):
    """internally used by function create_dataset for parallelization"""
    return data.loc[idx:(idx + obs), x_cols].as_matrix(), \
           data.loc[idx + obs, y_cols].as_matrix()


def munge_data(all_df, dropna=True):
    """load data and engineer"""

    def add_hist_data(df):
        """add previous x data points"""
        input_p_names = cfg.data_cfg['Input_param_names']
        shifted_df_list = []
        for shift in range(lookback):
            cols = ['{}_prev{}'.format(c, ~shift) for c in input_p_names]
            shifted_df = df.loc[:, input_p_names].shift(shift+1)
            shifted_df.columns = cols
            shifted_df_list.append(shifted_df)
        return [df, ] + shifted_df_list

    # make stationary by having differences only
    # all_df[orig_cols] = all_df[orig_cols].diff()
    # all_df.dropna(inplace=True)

    # normalize
    all_df[orig_cols] = scaler.fit_transform(all_df[orig_cols])

    # add lookback
    train_df_list = add_hist_data(all_df)
    all_df = pd.concat(train_df_list, axis=1)
    if dropna:
        all_df = all_df.iloc[lookback:, :]  # cutoff nans
    else:
        all_df.fillna(0, inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    # split train, test and validation set
    train_df = all_df[~all_df[p_id].isin(testset+valset)]
    test_df = all_df[all_df[p_id].isin(testset)]
    val_df = all_df[all_df[p_id].isin(valset)]

    if DEBUG:
        train_df = train_df.iloc[:n_debug, :]
        val_df = val_df.iloc[:n_debug, :]
        test_df = test_df.iloc[:n_debug, :]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def train_keras():
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, SimpleRNN
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.optimizers import SGD
    from keras import regularizers
    from keras.callbacks import EarlyStopping
    from keras import __version__ as keras_version

    from joblib import Parallel, delayed
    from multiprocessing import cpu_count

    print('Keras version: {}'.format(keras_version))
    batch_size = cfg.keras_cfg['batch_size']
    if DEBUG:
        assert batch_size < n_debug
    n_neurons = cfg.keras_cfg['n_neurons']
    n_epochs = cfg.keras_cfg['n_epochs']
    arch_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    arch_dict_cudnn = {'lstm': CuDNNLSTM, 'gru': CuDNNGRU, 'rnn': SimpleRNN}
    if gpu_available:
        ANN = arch_dict_cudnn[cfg.keras_cfg['arch']]
    else:
        ANN = arch_dict[cfg.keras_cfg['arch']]

    @measure_time
    def create_dataset(dataset, fit_to_batch=True):
        df_X = dataset.loc[:, x_cols]
        df_Y = dataset.loc[:, y_cols]

        if fit_to_batch:
            # cut the tail
            trunc_idx = len(dataset) % batch_size
            df_X = df_X.iloc[:-trunc_idx, :]
            df_Y = df_Y.iloc[:-trunc_idx, :]

            # reorder for batch training
            new_idx = np.tile(np.arange(batch_size), len(dataset) // batch_size)
            assert len(df_X)==new_idx.shape[0],\
                "{} != {}".format(len(df_X), new_idx.shape[0])
            df_X['new_idx'] = new_idx
            df_X.sort_values(by='new_idx', ascending=True, inplace=True)
            df_Y['new_idx'] = new_idx
            df_Y.sort_values(by='new_idx', ascending=True, inplace=True)

            df_X.reset_index(drop=True, inplace=True)
            df_Y.reset_index(drop=True, inplace=True)
            df_X = df_X[[c for c in df_X.columns if c != 'new_idx']]
            df_Y = df_Y[[c for c in df_Y.columns if c != 'new_idx']]

        x_mat = df_X.as_matrix()
        y_mat = df_Y.as_matrix()
        del df_X, df_Y

        x_mat = np.reshape(x_mat, (x_mat.shape[0], 1, x_mat.shape[1]))
        return x_mat, y_mat

    print("build dataset..")
    print('trainset..')
    X_tr, Y_tr = create_dataset(tra_df)
    print('valset..')
    X_val, Y_val = create_dataset(val_df)
    print('testset..')
    X_tst, Y_tst = create_dataset(tst_df)

    print('Shapes: train {}, val {}, test {}'.format(X_tr.shape,
                                                     X_val.shape,
                                                     X_tst.shape))

    # create model
    model = Sequential()
    model.add(
        ANN(n_neurons,
            # implementation=2,  # only known by non-CUDNN classes
            batch_input_shape=(batch_size, X_tr.shape[1], X_tr.shape[2]),
            kernel_regularizer=regularizers.l2(cfg.train_cfg['l2_reg_w']),
            activity_regularizer=regularizers.l2(cfg.train_cfg['l2_reg_w']),
            recurrent_regularizer=regularizers.l2(cfg.train_cfg['l2_reg_w']),
            stateful=True,
            ))
    model.add(Dropout(0.5))
    model.add(Dense(4))

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=cfg.train_cfg['early_stop_patience'],
                      verbose=0),
    ]

    # truncate dataset to have n samples dividable by batchsize for stateful RNN
    """trunc_idx = X_tr.shape[0] % batch_size
    X_tr = X_tr.iloc[:-trunc_idx, :]
    Y_tr = Y_tr.iloc[:-trunc_idx, :]"""

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_tr, Y_tr, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_val, Y_val), verbose=1,
                        shuffle=False,
                        callbacks=callbacks)
    return model.predict(X_tst, batch_size=batch_size), history.history


def tpotting():
    preds = []
    for target in output_p_names:
        tpot = TPOTRegressor(verbosity=2, cv=5, random_state=2017,
                             n_jobs=4,
                             periodic_checkpoint_folder='out/out_{}'.format(
                                 target))
        tpot.fit(tra_df[x_cols], tra_df[target])
        tpot.export('out/tpotted_{}.py'.format(target))
        preds.append(tpot.predict(tst_df[x_cols]))
    return pd.DataFrame({c: x for c, x in zip(output_p_names, preds)})


def train_linear(tra, tst, x_columns, y_columns):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import ElasticNetCV

    print('train')

    # lr_model = LinearRegression()
    preds = []
    for target in y_columns:
        lr_model = \
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False,
                                   interaction_only=False),
                PCA(iterated_power=4, svd_solver="randomized"),
                ElasticNetCV(l1_ratio=0.25, tol=0.001)
            )
        lr_model.fit(tra[x_columns], tra[target])
        preds.append(lr_model.predict(tst[x_columns]))
    return pd.DataFrame({c: x for c, x in zip(y_columns, preds)})


def plot_results(y, yhat):
    plt.plot(y)
    plt.plot(yhat)
    plt.show()


def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import pandas as pd
    from tpot import TPOTRegressor
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    os.system("taskset -p 0xffff %d" % os.getpid())  # reset core affinity

    # config
    gpu_available = len(get_available_gpus()) >= 1
    if cfg.debug_cfg['choose_debug_on_gpu_availability']:
        DEBUG = not gpu_available
    else:
        DEBUG = cfg.debug_cfg['DEBUG']
    n_debug = cfg.debug_cfg['n_debug']
    lookback = cfg.data_cfg['lookback']
    p_id = cfg.data_cfg['profile_id_colname']
    testset = cfg.data_cfg['testset']
    valset = cfg.data_cfg['valset']
    input_p_names = cfg.data_cfg['Input_param_names']
    output_p_names = cfg.data_cfg['Target_param_names']

    scaler = MinMaxScaler()
    dataset = pd.read_csv(join('input', 'measures.csv'), dtype=np.float32)

    orig_cols = [c for c in dataset.columns if c != p_id]

    tra_df, val_df, tst_df = munge_data(dataset.copy())
    # tra_df = pd.concat([tra_df, val_df], axis=0)

    # determine feature columns
    x_cols = []
    for col in tra_df.columns:
        for orig_input_param_name in input_p_names:
            if orig_input_param_name in col:
                x_cols.append(col)
    # determine target columns
    y_cols = output_p_names

    pred_df = pd.DataFrame(np.zeros(tst_df[orig_cols].shape),
                           columns=orig_cols)

    # tpot
    # yhat = tpotting()

    # linear model
    # yhat = train_linear(tra_df, tst_df, x_cols, y_cols)

    # keras
    yhat, hist = train_keras()
    prediction_start_idx = len(pred_df)-yhat.shape[0]
    pred_df.loc[prediction_start_idx:, y_cols] = yhat

    actual = \
        dataset[dataset[p_id].isin(testset)].loc[:, y_cols]
    actual.reset_index(drop=True, inplace=True)

    # untransform prediction
    inversed_pred = pd.DataFrame(
        scaler.inverse_transform(pred_df),
        columns=orig_cols).loc[:, y_cols]

    if DEBUG:
        actual = actual.iloc[:n_debug, :]
        inversed_pred = inversed_pred.iloc[:n_debug, :]
        print('actual {}, pred {}'.format(actual.shape, inversed_pred.shape))

    print('mse: {:.6} K²'.format(mean_squared_error(
        actual.iloc[prediction_start_idx:, :],
        inversed_pred.iloc[prediction_start_idx:, :])))

    # plots
    if cfg.plot_cfg['do_plot']:
        plt.subplot(211)
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.subplot(212)
        plot_results(actual, inversed_pred)

