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

# todo: xgb, lgbm attempts
# todo: taskset only if not 0xfff
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

        shifted_df_list = []
        for shift in range(lookback):
            cols = ['{}_lag{}'.format(c, ~shift) for c in df.columns]
            shifted_df = df.shift(shift+1)
            shifted_df.columns = cols
            shifted_df_list.append(shifted_df)
        return [df, ] + shifted_df_list

    input_cols = cfg.data_cfg['Input_param_names']

    # predict the logs
    all_df[y_cols] = np.sqrt(all_df[y_cols])

    # normalize y
    all_df[y_cols] = scaler_y.fit_transform(all_df[y_cols])

    # engineer new features
    if lookback == 1:
        train_feats = all_df.loc[:, input_cols]
        lag1 = train_feats.shift(1)
        lag1_diff = train_feats.diff()
        lag1_abs = abs(lag1_diff)
        lag1_sum = train_feats + lag1
        # change column names
        lag1.columns = ['{}_lag1'.format(c) for c in input_cols]
        lag1_diff.columns = ['{}_lag1_diff'.format(c) for c in input_cols]
        lag1_abs.columns = ['{}_lag1_abs'.format(c) for c in input_cols]
        lag1_sum.columns = ['{}_lag1_sum'.format(c) for c in input_cols]
        train_df_list = [all_df, lag1, lag1_diff, lag1_abs, lag1_sum]
    else:
        # add lookback
        train_df_list = add_hist_data(all_df.loc[:, input_cols])

    all_df = pd.concat(train_df_list, axis=1)

    if dropna:
        all_df.dropna(inplace=True)
    else:
        all_df.fillna(0, inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    # update x cols
    x_cols = [c for c in all_df.columns if c not in y_cols+[p_id,]]

    # normalize x
    scaler_x = MinMaxScaler()
    all_df[x_cols] = scaler_x.fit_transform(all_df[x_cols])
    logs = pd.DataFrame(np.log(all_df[x_cols]+1.5).values,
                        columns=['{}_ln'.format(c) for c in x_cols])
    all_df = pd.concat([all_df, logs], axis=1)

    scaler_x2 = MinMaxScaler()
    all_df[logs.columns] = scaler_x2.fit_transform(all_df[logs.columns])

    # last update of x cols
    x_cols = [c for c in all_df.columns if c not in y_cols + [p_id, ]]

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
            if cfg.keras_cfg['add_shifts']:
                # add shifts to dataset samples
                shifted_list_of_df_X = \
                    [df_X.shift(i+1) for i in range(batch_size-1)]
                df_X = pd.concat([df_X, ] + shifted_list_of_df_X, axis=0)
                df_X.dropna(inplace=True)
                df_X.reset_index(drop=True, inplace=True)

                shifted_list_of_df_Y = \
                    [df_Y.shift(i+1) for i in range(batch_size-1)]
                df_Y = pd.concat([df_Y, ] + shifted_list_of_df_Y, axis=0)
                df_Y.dropna(inplace=True)
                df_Y.reset_index(drop=True, inplace=True)

            # cut the tail
            trunc_idx = len(df_X) % batch_size
            df_X = df_X.iloc[:-trunc_idx, :]
            df_Y = df_Y.iloc[:-trunc_idx, :]

            # reorder for batch training
            new_idx = np.tile(np.arange(batch_size), len(df_X) // batch_size)
            assert len(df_X)==new_idx.shape[0],\
                "{} != {}".format(len(df_X), new_idx.shape[0])
            df_X['new_idx'] = new_idx
            df_X.sort_values(by='new_idx', ascending=True, inplace=True)
            df_Y['new_idx'] = new_idx
            df_Y.sort_values(by='new_idx', ascending=True, inplace=True)

            original_indices = [list(df_X.index), list(df_Y.index)]

            df_X.reset_index(drop=True, inplace=True)
            df_Y.reset_index(drop=True, inplace=True)
            df_X.drop(['new_idx', ], axis=1, inplace=True)
            df_Y.drop(['new_idx', ], axis=1, inplace=True)

        x_mat = df_X.as_matrix()
        y_mat = df_Y.as_matrix()
        del df_X, df_Y

        x_mat = np.reshape(x_mat, (x_mat.shape[0], 1, x_mat.shape[1]))
        return x_mat, y_mat, original_indices

    print("build dataset..")
    print('trainset..')
    X_tr, Y_tr, idx_tr = create_dataset(tra_df)
    print('valset..')
    X_val, Y_val, idx_val = create_dataset(val_df)
    print('testset..')
    X_tst, Y_tst, idx_tst = create_dataset(tst_df)

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

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_tr, Y_tr, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_val, Y_val), verbose=1,
                        shuffle=False,
                        callbacks=callbacks)
    model.reset_states()

    return model.predict(X_tst, batch_size=batch_size),\
           history.history, idx_tst


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


def train_linear():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import ElasticNetCV

    print('train')

    # todo: rewrite this..
    # lr_model = LinearRegression()
    preds = []
    for target in y_cols:
        lr_model = \
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False,
                                   interaction_only=False),
                PCA(iterated_power=4, svd_solver="randomized"),
                ElasticNetCV(l1_ratio=0.25, tol=0.001)
            )
        lr_model.fit(tra_df[x_cols], tra_df[target])
        preds.append(lr_model.predict(tst_df[x_cols]))
    return np.transpose(np.array(preds))


def train_extra_tree():
    from sklearn.ensemble import ExtraTreesRegressor

    print('train extra trees')
    et = ExtraTreesRegressor()
    et.fit(tra_df[x_cols], tra_df[y_cols])
    return et.predict(tst_df[x_cols])


def train_ridge():
    from sklearn.linear_model import Ridge
    print('train ridge')
    ridge = Ridge(alpha=40)
    ridge.fit(tra_df[x_cols], tra_df[y_cols])
    return ridge.predict(tst_df[x_cols])


def train_catboost():
    from catboost import CatBoostRegressor

    print('train catboost')
    preds = []
    for target in y_cols:
        cat = CatBoostRegressor()
        cat.fit(tra_df[x_cols], tra_df[target])
        preds.append(cat.predict(tst_df[x_cols]))
    return np.transpose(np.array(preds))


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
    y_cols = cfg.data_cfg['Target_param_names']
    batch_size = cfg.keras_cfg['batch_size']

    scaler_y = MinMaxScaler()
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

    # prepare target data
    actual = \
        dataset[dataset[p_id].isin(testset)].loc[:, y_cols]
    actual.reset_index(drop=True, inplace=True)
    if DEBUG:
        actual = actual.iloc[:n_debug, :]

    # keras
    if cfg.keras_cfg['do_train']:
        yhat, hist, tst_idx = train_keras()
        # untransform actual
        yhat = pd.DataFrame(yhat, columns=y_cols, index=tst_idx).sort_index()
        yhat = yhat.values

    # tpot
    # yhat = tpotting()

    # linear model
    #yhat = train_linear()

    # extra trees
    #yhat = train_extra_tree()

    # catboost
    #yhat = train_catboost()

    # ridge
    yhat = train_ridge()
    #yhat = (yhat + yhat_ridge[:yhat.shape[0], :])/2

    inversed_pred = pd.DataFrame(scaler_y.inverse_transform(yhat),
                                 columns=y_cols)
    inversed_pred = np.square(inversed_pred)
    print('mse: {:.6} K²'.format(mean_squared_error(actual.values[
                                                    :yhat.shape[0], :],
                                                    inversed_pred)))

    # plots
    if cfg.plot_cfg['do_plot']:
        if cfg.keras_cfg['do_train']:
            plt.subplot(211)
            plt.plot(hist['loss'])
            plt.plot(hist['val_loss'])
        plt.subplot(212)
        plot_results(actual, inversed_pred)

