import sqlite3
import pandas as pd
import numpy as np
from os.path import join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import config as cfg
from kirgsn.data import DataManager

ID_TO_READ = 'b9f735'


def plot_results(y, yhat):
    plt.plot(y, alpha=0.6, color='darkorange')
    plt.plot(yhat, lw=2, color='navy')
    plt.show()


if __name__ == '__main__':

    with sqlite3.connect(cfg.data_cfg['db_path']) as con:
        query = """SELECT * FROM predictions WHERE id=?"""
        yhat = pd.read_sql_query(query, con, params=(ID_TO_READ,))

    yhat.drop(['id', 'idx'], inplace=True, axis=1)
    dm = DataManager(join('input', 'measures.csv'))
    actual = dm.actual

    # trunc actual if prediction is shorter
    if yhat.shape[0] != actual.shape[0]:
        print('trunc actual from {} to {} samples'.format(actual.shape[0],
                                                          yhat.shape[0]))
        offset = actual.shape[0] - yhat.shape[0]
        actual = actual.iloc[offset:, :]
    print('mse: {:.6} K²'.format(mean_squared_error(actual.values,
                                                    yhat.values)))
    plot_results(actual, yhat)
