"""Author: WKirgsn, 2017"""
import warnings
import multiprocessing
import time
import os
from os.path import join

import numpy as np
from tensorflow.python.client import device_lib

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
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import ElasticNetCV, LogisticRegression

    print('train')

    preds = []
    for target in y_cols:
        """lr_model = \
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False,
                                   interaction_only=False),
                PCA(iterated_power=4, svd_solver="randomized"),
                ElasticNetCV(l1_ratio=0.25, tol=0.001)
            )"""
        lr_model = ElasticNetCV()
        lr_model.fit(tra_df[x_cols], tra_df[target])
        preds.append(lr_model.predict(tst_df[x_cols]))
    return np.transpose(np.array(preds))


def train_extra_tree(dm):
    from sklearn.ensemble import ExtraTreesRegressor

    print('train extra trees')
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    et = ExtraTreesRegressor()
    et.fit(tra_df[x_cols], tra_df[dm.y_cols])
    return et.predict(tst_df[x_cols])


def train_ridge(dm):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    print('train ridge')
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    ridge = make_pipeline(
        #PolynomialFeatures(degree=2, include_bias=False,
         #                  interaction_only=True),
        Ridge(alpha=50000)
    )

    ridge.fit(tra_df[x_cols], tra_df[dm.y_cols])
    return ridge.predict(tst_df[x_cols])


def train_catboost(dm):
    from catboost import CatBoostRegressor

    print('train catboost')
    preds = []
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    for target in dm.y_cols:
        cat = CatBoostRegressor()
        cat.fit(tra_df[x_cols], tra_df[target])
        preds.append(cat.predict(tst_df[x_cols]))
    return np.transpose(np.array(preds))


def train_SVR(dm):
    from sklearn.svm import SVR

    preds = []
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    for target in dm.y_cols:
        cat = SVR(C=1.0)
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

    from kirgsn.data import DataManager

    os.system("taskset -p 0xffff %d" % os.getpid())  # reset core affinity

    # config
    gpu_available = len(get_available_gpus()) >= 1
    if cfg.debug_cfg['choose_debug_on_gpu_availability']:
        DEBUG = not gpu_available
    else:
        DEBUG = cfg.debug_cfg['DEBUG']
    n_debug = cfg.debug_cfg['n_debug']

    batch_size = cfg.keras_cfg['batch_size']

    dm = DataManager(join('input', 'measures.csv'))
    dm.predict_transformed_targets(np.sqrt, np.square)
    dm.normalize_targets()
    dm.add_stats_from_hist_data(lookback=10)
    dm.add_lag_feats()
    dm.normalize_features(dm.scaler_x_1)
    dm.add_transformed_feats(np.log, 'ln')
    dm.normalize_features(dm.scaler_x_2)

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
    #yhat = train_extra_tree(dm=dm)

    # catboost
    #yhat = train_catboost(dm=dm)

    # ridge
    #yhat = train_ridge(dm=dm)

    # SVR
    yhat = train_SVR(dm=dm)

    actual = dm.actual
    inversed_pred = dm.inverse_prediction(yhat)
    # trunc actual if prediction is shorter
    if yhat.shape[0] != actual.shape[0]:
        print('trunc actual from {} to {} samples'.format(actual.shape[0],
                                                          yhat.shape[0]))
        actual = actual.iloc[:yhat.shape[0], :]
    print('mse: {:.6} K²'.format(mean_squared_error(actual.values,
                                                    inversed_pred)))

    # plots
    if cfg.plot_cfg['do_plot']:
        if cfg.keras_cfg['do_train']:
            plt.subplot(211)
            plt.plot(hist['loss'])
            plt.plot(hist['val_loss'])
            plt.subplot(212)
        plot_results(actual, inversed_pred)

