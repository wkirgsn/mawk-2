
from os.path import join
import numpy as np
import warnings
import multiprocessing
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


def munge_data(all_df, dropna=True):
    """load data for linear regressors"""
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


def get_norm_stats(all):
    temp = all[0].copy()
    for i in range(1, len(all)):
        temp.append(all[i], ignore_index=True)
    return temp.mean(), temp.max(), temp.min()


def train_keras():
    from keras.models import Sequential
    from keras.layers import LSTM, GRU
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping
    from keras import __version__ as keras_version

    print('Keras version: {}'.format(keras_version))
    batch_size = 64
    n_neurons = 8
    n_epochs = 500
    observation_len = 10

    def create_dataset(dataset, observe=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - observe):
            dataX.append(dataset.loc[i:(i + observe), x_cols])
            dataY.append(dataset.loc[i + observe, y_cols])
        return np.array([p.as_matrix() for p in dataX]), \
               np.array([p.as_matrix() for p in dataY])

    X_tr, Y_tr = create_dataset(tra_df, observation_len-1)
    X_val, Y_val = create_dataset(val_df, observation_len-1)
    X_tst, Y_tst = create_dataset(tst_df, observation_len-1)

    # reshape for keras (not necessary here)
    # X_tr = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
    # X_tst = np.reshape(X_tst, (X_tst.shape[0], 1, X_tst.shape[1]))
    print('n samples = {}'.format(X_tr.shape[0]))

    # create model
    model = Sequential()
    model.add(LSTM(n_neurons,
                   batch_input_shape=(batch_size,
                                      X_tr.shape[1],
                                      X_tr.shape[2])))
    model.add(Dense(4))

    """callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ]"""

    # truncate dataset to have n samples dividable by batchsize for stateful RNN
    """trunc_idx = X_tr.shape[0] % batch_size
    X_tr = X_tr.iloc[:-trunc_idx, :]
    Y_tr = Y_tr.iloc[:-trunc_idx, :]"""

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_tr, Y_tr, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_val, Y_val), verbose=1,
                        shuffle=False)
    return model.predict(X_tst)


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

    import pandas as pd
    from tpot import TPOTRegressor
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn
    from sklearn.preprocessing import MinMaxScaler

    from sklearn.model_selection import train_test_split

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
    yhat = train_keras()
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

    print('mse: {} K²'.format(mean_squared_error(
        actual.iloc[prediction_start_idx:, :],
        inversed_pred.iloc[prediction_start_idx:, :])))

    # plots
    #plot_results(actual, inversed_pred)

