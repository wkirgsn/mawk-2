import argparse

import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing.config as cfg
from preprocessing.data import DataManager
from preprocessing import file_utils as futils


def plot_results(y, yhat):
    plt.subplot(211)
    plt.plot(y, alpha=0.6, color='darkorange')
    plt.plot(yhat, lw=2, color='navy')
    plt.xlabel('time in s')
    plt.ylabel('temperature in K')
    plt.subplot(212)
    plt.plot(yhat-y, color='red')
    plt.xlabel('time in s')
    plt.ylabel('temperature in K')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize performance of the '
                                                 'given model uid.')
    parser.add_argument('model_uid',
                        help='The 6-digit model uid in hex')
    args = parser.parse_args()

    with sqlite3.connect(cfg.data_cfg['db_path']) as con:
        query = """SELECT * FROM predictions WHERE id=?"""
        yhat = pd.read_sql_query(query, con, params=(args.model_uid,))

    yhat.drop(['id', 'idx'], inplace=True, axis=1)

    dm = DataManager(cfg.data_cfg['file_path'])
    actual = futils.truncate_actual_target(dm.actual, yhat)

    print('mse: {:.6} K²'.format(mean_squared_error(actual.values,
                                                    yhat.values)))
    try:
        plot_results(actual, yhat)
    except Exception:
        print('plot failed')
