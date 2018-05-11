"""Author: WKirgsn, 2017"""
import warnings
from os.path import join
import sqlite3
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import validation_curve, TimeSeriesSplit
import lightgbm

from preprocessing.data import DataManager, ReSamplerForBatchTraining
import preprocessing.config as cfg
import preprocessing.file_utils as futils

warnings.filterwarnings("ignore")
np.random.seed(2017)


def tpotting():
    from tpot import TPOTRegressor
    """deprecated"""
    for target in range(4):  # replace with real targets
        tpot = TPOTRegressor(verbosity=2, cv=5, random_state=2017,
                             n_jobs=4,
                             periodic_checkpoint_folder='out/out_{}'.format(
                                 target))
        tpot.fit(tra_df['x_cols'], tra_df[target])
        tpot.export('out/tpotted_{}.py'.format(target))


def plot_val_curves(train_scores, test_scores, param_range):
    """ validation curves

    Usage:
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


def _train(_model, is_mimo=True, with_val_set=True):
    train_d = {'X': tra_df[dm.cl.x_cols],
               'y': tra_df[dm.cl.y_cols],
               }

    if with_val_set:
        train_d['eval_set'] = (val_df.loc[:, dm.cl.x_cols],
                               val_df.loc[:, dm.cl.y_cols])
        train_d['early_stopping_rounds'] = 30

    if is_mimo:
        print('start training...')
        _model.fit(**train_d)
        ret = _model.predict(tst_df[dm.cl.x_cols])
    else:
        ret = []
        for t in dm.cl.y_cols:
            print('start training against {}'.format(t))
            train_d['y'] = tra_df.loc[:, t]
            if with_val_set:
                train_d['eval_set'] = \
                    (val_df.loc[:, dm.cl.x_cols],
                     val_df.loc[:, t])
            _model.fit(**train_d)
            ret.append(_model.predict(tst_df[dm.cl.x_cols]).reshape((-1, 1)))
        ret = np.hstack(ret)
    return ret


if __name__ == '__main__':

    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    # featurize dataset (feature engineering)
    dm = DataManager(cfg.data_cfg['file_path'])
    tra_df, val_df, tst_df = dm.get_featurized_sets()

    model = lightgbm.LGBMRegressor(**cfg.lgbm_cfg['params_found_by_skopt'])
    tscv = TimeSeriesSplit()

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
    futils.save_predictions(model_uuid, inversed_pred)

    # plots
    if cfg.plot_cfg['do_plot']:
        try:
            plot_results(actual, inversed_pred)
        except Exception:
            print('Plot failed...')
