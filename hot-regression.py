"""Author: WKirgsn, 2017"""
import warnings
import multiprocessing
import time
import os
from os.path import join
import sqlite3
import uuid

import numpy as np
from tensorflow.python.client import device_lib

import config as cfg

warnings.filterwarnings("ignore")
np.random.seed(2017)

# todo: xgb, lgbm attempts
# todo: save train trend history for visualizing
# todo: apply bayesian opt for hyperparams


def _prll_create_dataset(idx, obs, data, x_cols, y_cols):
    """deprecated.
    internally used by function create_dataset for parallelization"""
    return data.loc[idx:(idx + obs), x_cols].as_matrix(), \
           data.loc[idx + obs, y_cols].as_matrix()


def build_keras_model(x_shape=(100, 1, 10)):
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, SimpleRNN
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.optimizers import Adam, Nadam
    from keras import regularizers
    from keras import __version__ as keras_version

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


def tpotting():
    """deprecated"""
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


def train_extra_tree(dm):
    from sklearn.ensemble import ExtraTreesRegressor

    print('train extra trees')
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    et = ExtraTreesRegressor()
    et.fit(tra_df[x_cols], tra_df[dm.y_cols])
    return et.predict(tst_df[x_cols])


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
    print('Train SVR')
    preds = []
    tra_df = dm.tra_df
    tst_df = dm.tst_df
    x_cols = dm.x_cols
    for target in dm.y_cols:
        cat = SVR(C=1.0)
        cat.fit(tra_df[x_cols], tra_df[target])
        preds.append(cat.predict(tst_df[x_cols]))
    return np.transpose(np.array(preds))


def plot_val_curves(train_scores, test_scores, param_range):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("alpha")
    plt.ylabel("Score MSE")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def plot_results(y, yhat):
    plt.plot(y, alpha=0.6, color='darkorange')
    plt.plot(yhat, lw=2, color='navy')
    plt.show()


def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import pandas as pd
    #from tpot import TPOTRegressor
    from sklearn.metrics import mean_squared_error, make_scorer
    import matplotlib.pyplot as plt
    import seaborn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, Lasso, LassoLarsCV, ElasticNet
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
    from sklearn.model_selection import validation_curve, TimeSeriesSplit
    from sklearn.feature_selection import SelectFromModel, SelectPercentile, \
        f_regression
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.tree import ExtraTreeRegressor
    from catboost import CatBoostRegressor
    import lightgbm
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping, LearningRateScheduler

    from kirgsn.data import DataManager, ReSamplerForBatchTraining

    # os.system("taskset -p 0xffff %d" % os.getpid())  # reset core affinity

    # config
    gpu_available = len(get_available_gpus()) >= 1
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
    model_pool = {'': ''}  # baustelle

    dm = DataManager(join('input', 'measures.csv'))

    # featurize dataset (feature engineering)
    tra_df, val_df, tst_df = dm.get_featurized_sets()

    if cfg.keras_cfg['do_train']:

        class CustomKerasRegressor(KerasRegressor):
            def reset_states(self):
                self.model.reset_states()

        def reshape_input_for_batch_train(x_set):
            assert isinstance(x_set, pd.DataFrame)
            x = x_set.as_matrix()
            return np.reshape(x, (x.shape[0], 1, x.shape[1]))

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

    else:
        # build pipeline
        model = lightgbm.LGBMRegressor(n_estimators=1000)
        """ validation curves
        param_range = np.linspace(1, 10, num=10)
        tscv = TimeSeriesSplit(n_splits=3)
        tra_scores, tst_scores = validation_curve(pipe,
                                                  tra_df[dm.cl.x_cols],
                                                  tra_df[dm.cl.y_cols],
                                                  param_name='extratreesregressor__max_depth',
                                                  param_range=param_range,
                                                  cv=tscv,
                                                  scoring=
                                                  make_scorer(mean_squared_error),
                                                  verbose=5, n_jobs=2
                                                  )
        plot_val_curves(tra_scores, tst_scores, param_range)
        """
        def _train(_model, is_mimo=True, with_val_set=True):
            train_d = {'X': tra_df[dm.cl.x_cols],
                       'y': tra_df[dm.cl.y_cols]}
            if with_val_set:
                train_d['eval_set'] = (val_df[dm.cl.x_cols],
                                       val_df[dm.cl.y_cols])
                train_d['early_stopping_rounds'] = 30

            if is_mimo:
                print('start training...')
                _model.fit(**train_d)
                ret = _model.predict(tst_df[dm.cl.x_cols])
            else:
                ret = []
                for t in dm.cl.y_cols:
                    print('start training against {}'.format(t))
                    train_d['y'] = tra_df[t]
                    if with_val_set:
                        train_d['eval_set'] = (val_df[dm.cl.x_cols], val_df[t])
                    _model.fit(**train_d)
                    ret.append(
                        _model.predict(tst_df[dm.cl.x_cols]).reshape((-1, 1)))
                ret = np.hstack(ret)
            return ret

        yhat = _train(model, is_mimo=False)

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
    if cfg.data_cfg['save_predictions']:
        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            # create table if not exists
            query = """CREATE TABLE IF NOT EXISTS 
                predictions(id text, idx int, {} real, {} real, {} real, 
                {} real)""".format(*inversed_pred.columns)
            con.execute(query)

            # format prediction
            df_to_db = inversed_pred.copy()
            df_to_db['id'] = model_uuid
            df_to_db['idx'] = inversed_pred.index
            entries = [tuple(x) for x in np.roll(df_to_db.values,
                                                 shift=2, axis=1)]
            con.executemany('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)',
                            entries)
            print('Predictions of model with uuid {} saved to db.'.format(
                model_uuid))
    # plots
    if cfg.plot_cfg['do_plot']:
        if cfg.keras_cfg['do_train']:
            plt.subplot(211)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.subplot(212)
        plot_results(actual, inversed_pred)
