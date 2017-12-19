
from os.path import join
import numpy as np
import warnings
import multiprocessing
import time
import os

warnings.filterwarnings("ignore")
np.random.seed(2017)

DEBUG = False
n_debug = 50  # first n timestamps to use if debug
Input_param_names = ['ambient',
                     'coolant',
                     'u_d',
                     'u_q',
                     'motor_speed',
                     'torque',
                     'i_d',
                     'i_q']

Target_param_names = ['pm',
                      'stator_yoke',
                      'stator_tooth',
                      'stator_winding']

profile_id_colname = 'profile_id'
lookback = 2

valset = ['31', ]
testset = ['20', ]
loadsets = ['4', '6', '10', '11', '20', '27', '29', '30',
            '31', '32', '36']
file_path = "/home/wilhelmk/Messdaten/PMSM_Lastprofile/hdf/all_load_profiles"


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
            cols = ['{}_prev{}'.format(c, ~shift) for c in Input_param_names]
            shifted_df = df.loc[:, Input_param_names].shift(shift+1)
            shifted_df.columns = cols
            shifted_df_list.append(shifted_df)
        return [df, ] + shifted_df_list


    # make stationary by having differences only
    #all_df[orig_cols] = all_df[orig_cols].diff()
    #all_df.dropna(inplace=True)
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
    train_df = all_df[~all_df[profile_id_colname].isin(testset+valset)]
    test_df = all_df[all_df[profile_id_colname].isin(testset)]
    val_df = all_df[all_df[profile_id_colname].isin(valset)]

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
    from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping
    from keras import __version__ as keras_version

    from joblib import Parallel, delayed
    from multiprocessing import cpu_count

    print('Keras version: {}'.format(keras_version))
    batch_size = 128
    n_neurons = 64
    n_epochs = 200
    observation_len = 50

    @measure_time
    def create_dataset(dataset, observe=1):
        # full cpu usage does not work somehow
        n_cpu = 3  # maximum that works
        dataXY = Parallel(n_jobs=n_cpu)\
            (delayed(_prll_create_dataset)(i, observe, dataset, x_cols,
                                           y_cols)
             for i in range(len(dataset) - observe))
        separated = list(zip(*dataXY))
        return np.array(separated[0]), np.array(separated[1])

    print("build dataset..")
    print('trainset..')
    X_tr, Y_tr = create_dataset(tra_df, observation_len-1)
    print('valset..')
    X_val, Y_val = create_dataset(val_df, observation_len-1)
    print('testset..')
    X_tst, Y_tst = create_dataset(tst_df, observation_len-1)

    # reshape for keras (not necessary here)
    # X_tr = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
    # X_tst = np.reshape(X_tst, (X_tst.shape[0], 1, X_tst.shape[1]))
    print('n samples = {}'.format(X_tr.shape[0]))

    # create model
    model = Sequential()
    model.add(
        CuDNNLSTM(n_neurons,
                  # implementation=2,  # only known by non-CUDNN classes
                  input_shape=(X_tr.shape[1], X_tr.shape[2])
                  ))
    model.add(Dense(4))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
    ]

    # truncate dataset to have n samples dividable by batchsize for stateful RNN
    """trunc_idx = X_tr.shape[0] % batch_size
    X_tr = X_tr.iloc[:-trunc_idx, :]
    Y_tr = Y_tr.iloc[:-trunc_idx, :]"""

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_tr, Y_tr, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_val, Y_val), verbose=1,
                        shuffle=True,
                        callbacks=callbacks)
    return model.predict(X_tst), history.history


def tpotting():
    preds = []
    for target in Target_param_names:
        tpot = TPOTRegressor(verbosity=2, cv=5, random_state=2017,
                             n_jobs=4,
                             periodic_checkpoint_folder='out/out_{}'.format(
                                 target))
        tpot.fit(tra_df[x_cols], tra_df[target])
        tpot.export('out/tpotted_{}.py'.format(target))
        preds.append(tpot.predict(tst_df[x_cols]))
    return pd.DataFrame({c: x for c, x in zip(Target_param_names, preds)})


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


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import joblib
    import pandas as pd
    from tpot import TPOTRegressor
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    os.system("taskset -p 0xffffffffff %d" % os.getpid())  # reset core affinity

    scaler = MinMaxScaler()
    dataset = pd.read_csv(join('input', 'measures.csv'), dtype=np.float32)

    orig_cols = [c for c in dataset.columns if c != profile_id_colname]

    tra_df, val_df, tst_df = munge_data(dataset.copy())
    # tra_df = pd.concat([tra_df, val_df], axis=0)
    x_cols = [col for input in Input_param_names for col in tra_df.columns if
              col[:3] in input]
    y_cols = Target_param_names

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
        dataset[dataset[profile_id_colname].isin(testset)].loc[:, y_cols]
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
    """plt.subplot(211)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.subplot(212)
    plot_results(actual, inversed_pred)"""

