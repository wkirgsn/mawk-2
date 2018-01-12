"""
Author: Kirgsn, 2017, https://www.kaggle.com/wkirgsn
"""
import numpy as np
import pandas as pd

import config as cfg
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnManager:
    def __init__(self, df):
        self.original = df.columns
        self._x, self._y = None, cfg.data_cfg['Target_param_names']
        self.update(df)
        self.stash = []

    @property
    def x_cols(self):
        return self._x

    @x_cols.setter
    def x_cols(self, value):
        self._x = value

    @property
    def y_cols(self):
        return self._y

    def update(self, df, stash=None):
        x_cols = []
        for col in df.columns:
            for p in cfg.data_cfg['Input_param_names']:
                if p in col:
                    x_cols.append(col)
        if stash is not None:
            assert isinstance(stash, list)
            x_cols.extend(stash)
            self.empty_stash()
        self.x_cols = x_cols

    def stash(self, cols):
        self.stash.extend(cols)

    def empty_stash(self):
        self.stash = []


class DataManager:

    PROFILE_ID_COL = 'profile_id'

    def __init__(self, path):
        # original data
        self.dataset = pd.read_csv(path, dtype=np.float32)
        # downsample
        #self.dataset = self.dataset.iloc[::2, :]
        # drop profiles
        """drop_p = ['11', ]
        self.dataset.drop(axis=1, inplace=True, index=self.dataset[self.dataset[
            self.PROFILE_ID_COL].isin(drop_p)].index)"""
        # feature engineered dataset
        self.df = self.dataset.copy()

        # column management
        self.cl = ColumnManager(self.df)
        self.cl.x_cols = cfg.data_cfg['Input_param_names']

        # build pipeline building blocks

        featurize_union = FeatureUnion([('simple_trans_y',
                                        SimpleTransformer(np.sqrt,
                                                          np.square,
                                                          self.cl.y_cols
                                                          )),
                                       ('lag_feats_x',
                                        LagFeatures(self.cl.x_cols)),
                                       ('rolling_feats_x',
                                        RollingFeatures(self.cl.x_cols))
                                       ])

        featurize_pipe = FeatureUnionReframer.make_df_retaining(featurize_union)

        scaling_union = FeatureUnion([('scaler_x', Scaler(StandardScaler(),
                                                          self.cl,
                                                          select='x')),
                                     ('scaler_y', Scaler(StandardScaler(),
                                                         self.cl, select='y'))
                                     ])
        scaling_pipe = FeatureUnionReframer.make_df_retaining(scaling_union)

        self.pipe = Pipeline([
            ('feat_engineer', featurize_pipe),
            ('cleaning', DFCleaner()),
            ('scaler', scaling_pipe),
            ('ident', IdentityEstimator())
        ])

    @property
    def tra_df(self):
        sub_df = self.df[~self.df[self.PROFILE_ID_COL].isin(
            cfg.data_cfg['testset']# + cfg.data_cfg['valset']
        )]
        sub_df.reset_index(drop=True, inplace=True)
        self.cl.update(sub_df)
        return sub_df

    @property
    def val_df(self):
        sub_df = self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg['valset'])]
        sub_df.reset_index(drop=True, inplace=True)
        return sub_df

    @property
    def tst_df(self):
        sub_df = self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg[
                                                             'testset'])]
        sub_df.reset_index(drop=True, inplace=True)
        return sub_df

    @property
    def actual(self):
        sub_df = self.dataset[self.dataset[self.PROFILE_ID_COL].isin(cfg.data_cfg[
                                                            'testset'])]
        return sub_df[self.cl.y_cols].reset_index(drop=True)

    def get_featurized_sets(self):
        tra_df = self.tra_df
        tst_df = self.tst_df
        val_df = self.val_df

        tra_df = self.pipe.fit_transform(tra_df)
        tst_df = self.pipe.transform(tst_df)
        if val_df is not None:
            val_df = self.pipe.transform(val_df)

        self.cl.update(tra_df)
        return tra_df, val_df, tst_df

    def inverse_prediction(self, pred):
        simple_transformer = {k: v for k, v in self.pipe.named_steps[
            'feat_engineer'].named_steps['union'].transformer_list}['simple_trans_y']

        scaler = {k: v for k, v in self.pipe.named_steps[
            'scaler'].named_steps['union'].transformer_list}['scaler_y']

        reduced_pipe = make_pipeline(simple_transformer, scaler)

        inversed = pd.DataFrame(reduced_pipe.inverse_transform(pred),
                                columns=self.cl.y_cols)
        return inversed

    def plot(self):
        from pandas.plotting import autocorrelation_plot
        import matplotlib.pyplot as plt
        self.df[[c for c in self.x_cols if 'rolling' in c] +
                self.y_cols].plot(subplots=True, sharex=True)
        plt.show()


class SimpleTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""
    def __init__(self, trans_func, untrans_func, columns):
        self.transform_func = trans_func
        self.inverse_transform_func = untrans_func
        self.cols = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = self._get_selection(x)
        return self.transform_func(x)

    def inverse_transform(self, x):
        return self.inverse_transform_func(x)

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.cols


class LagFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds arithmetic variations between current and lag_x
    observation"""
    def __init__(self, columns, lookback=1):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        dfs = []
        for lback in range(1, self.lookback + 1):
            lag_feats = {'lag{}': X.shift(lback),
                         'lag{}_diff': X.diff(periods=lback),
                         }
            lag_feats['lag{}_abs'] = abs(lag_feats['lag{}_diff'])
            lag_feats['lag{}_sum'] = X + lag_feats['lag{}']

            lag_feats = {key.format(lback): value for key, value
                         in lag_feats.items()}
            # update columns
            for k in lag_feats:
                lag_feats[k].columns = ['{}_{}'.format(c, k) for c in
                                        X.columns]

            dfs.append(pd.concat(list(lag_feats.values()), axis=1))
        df = pd.concat(dfs, axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class RollingFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds rolling statistics"""
    def __init__(self, columns, lookback=10):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        feat_d = {'std': X.rolling(self.lookback).std(),
                  'mean': X.rolling(self.lookback).mean(),
                  # 'sum': X.rolling(self.lookback).sum()
                  }
        for k in feat_d:
            feat_d[k].columns = \
                ['{}_rolling{}_{}'.format(c, self.lookback, k) for
                 c in X.columns]
        df = pd.concat(list(feat_d.values()), axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class DFCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X.dropna(inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X


class IdentityEstimator(BaseEstimator, TransformerMixin):
    """This class is for replacing a basic identity estimator with one that
    returns the full input pandas DataFrame instead of a numpy arr
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Scaler(BaseEstimator, TransformerMixin):
    """scales selected columns only with given scaler.
    Parameter 'select' is either 'x' or 'y' """
    def __init__(self, scaler, column_manager, select='x'):
        self.cl = column_manager
        self.scaler = scaler
        self.select = select
        self.cols = []

    def fit(self, X, y=None):
        X = self.get_selection(X)
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X[self.cols]
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        if self.select.lower() == 'x':
            self.cl.update(df)
            self.cols = self.cl.x_cols
        elif self.select.lower() == 'y':
            self.cols = self.cl.y_cols
        else:
            raise NotImplementedError()
        return df[self.cols]

    def get_feature_names(self):
        return self.cols


class FeatureUnionReframer(BaseEstimator, TransformerMixin):
    """Transforms preceding FeatureUnion's output back into Dataframe"""
    def __init__(self, feat_union, cutoff_transformer_name=True):
        self.union = feat_union
        self.cutoff_transformer_name = cutoff_transformer_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        if self.cutoff_transformer_name:
            cols = [c.split('__')[1] for c in self.union.get_feature_names()]
        else:
            cols = self.union.get_feature_names()
        df = pd.DataFrame(data=X, columns=cols)
        return df

    @classmethod
    def make_df_retaining(cls, feature_union):
        """With this method a feature union will be returned as a pipeline
        where the first step is the union and the second is a transformer that
        re-applies the columns to the union's output"""
        return Pipeline([('union', feature_union),
                         ('reframe', cls(feature_union))])


