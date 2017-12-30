"""
Author: Kirgsn, 2017, https://www.kaggle.com/wkirgsn
"""
import numpy as np
import pandas as pd

import config as cfg
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataManager:

    PROFILE_ID_COL = 'profile_id'

    def __init__(self, path):
        # scalers
        self.scaler_x_1 = MinMaxScaler()
        self.scaler_x_2 = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # original data
        self.dataset = pd.read_csv(path, dtype=np.float32)
        # feature engineered dataset
        self.df = self.dataset.copy()

        # target transformation
        self.target_untrans_func = None

        # column management
        self.orig_cols = [c for c in self.dataset.columns if
                          c != self.PROFILE_ID_COL]
        self._x_cols = cfg.data_cfg['Input_param_names']
        self.y_cols = cfg.data_cfg['Target_param_names']

    @property
    def x_cols(self):
        if self.tra_df is not None:
            self._x_cols = []
            for col in self.tra_df.columns:
                for p in cfg.data_cfg['Input_param_names']:
                    if p in col:
                        self._x_cols.append(col)
        return self._x_cols

    @property
    def tra_df(self):
        sub_df = self.df[~self.df[self.PROFILE_ID_COL].isin(
            cfg.data_cfg['testset'] + cfg.data_cfg['valset'])]
        return sub_df.reset_index(drop=True)

    @property
    def val_df(self):
        sub_df = self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg['valset'])]
        return sub_df.reset_index(drop=True)

    @property
    def tst_df(self):
        sub_df = self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg[
                                                             'testset'])]
        return sub_df.reset_index(drop=True)

    @property
    def actual(self):
        sub_df = self.dataset[self.dataset[self.PROFILE_ID_COL].isin(cfg.data_cfg[
                                                            'testset'])]
        return sub_df[self.y_cols].reset_index(drop=True)

    def dropna(self):
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def predict_transformed_targets(self, trans_func, untransform_func):
        """transforms target variables with given transformation function"""
        self.df[self.y_cols] = trans_func(self.df.loc[:, self.y_cols])
        self.target_untrans_func = untransform_func

    def normalize_targets(self):
        self.df[self.y_cols] = \
            self.scaler_y.fit_transform(self.df[self.y_cols])

    def normalize_features(self, scaler):
        x_cols = self.x_cols
        self.df[x_cols] = scaler.fit_transform(self.df[x_cols])

    def add_lag_feats(self, lookback=1):
        input_cols = self.x_cols
        train_feats = self.df.loc[:, input_cols]
        lag_feats = {'lag{}': train_feats.shift(lookback),
                     'lag{}_diff': train_feats.diff(),
                     }
        lag_feats['lag{}_abs'] = abs(lag_feats['lag{}_diff'])
        lag_feats['lag{}_sum'] = train_feats + lag_feats['lag{}']

        lag_feats = {key.format(lookback): value for key, value
                     in lag_feats.items()}
        # update columns
        for k in lag_feats:
            lag_feats[k].columns = ['{}_{}'.format(c, k) for c in input_cols]
        # add to dataset
        self.df = pd.concat([self.df, ] + list(lag_feats.values()), axis=1)
        self.dropna()

    def add_transformed_feats(self, trans_func, lbl):
        """transform the current features with given transformation function"""
        x_cols = self.x_cols
        trans_df = pd.DataFrame(trans_func(self.df[x_cols] + 1.5).values,
                                columns=['{}_{}'.format(c, lbl) for c
                                         in x_cols])
        self.df = pd.concat([self.df, trans_df], axis=1)

    def add_stats_from_hist_data(self, lookback=10):
        """add previous x data point statistics"""
        cols = self.x_cols
        feat_d = {'std': self.df[cols].rolling(lookback).std(),
                  'mean': self.df[cols].rolling(lookback).mean(),
                  'sum': self.df[cols].rolling(lookback).sum()}
        for k in feat_d:
            feat_d[k].columns = ['{}_rolling{}_{}'.format(c, lookback, k) for
                                 c in cols]
        self.df = pd.concat([self.df, ] + list(feat_d.values()), axis=1)
        self.dropna()

    def inverse_prediction(self, pred):
        inversed = pd.DataFrame(self.scaler_y.inverse_transform(pred),
                                columns=self.y_cols)
        if self.target_untrans_func is not None:
            inversed = self.target_untrans_func(inversed)
        return inversed
