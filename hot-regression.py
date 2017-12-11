
from os.path import join
import numpy as np
import warnings
import multiprocessing
warnings.filterwarnings("ignore")
np.random.seed(2017)

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

valset = ['31', ]
testset = ['20', ]
loadsets = ['4', '6', '10', '11', '20', '27', '29', '30',
            '31', '32', '36']
file_path = "/home/wilhelmk/Messdaten/PMSM_Lastprofile/hdf/all_load_profiles"
subsequence_len = 100


def munge_data(all_df, lookback=5, dropna=True):
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

    return train_df, val_df, test_df


def get_norm_stats(all):
    temp = all[0].copy()
    for i in range(1, len(all)):
        temp.append(all[i], ignore_index=True)
    return temp.mean(), temp.max(), temp.min()


def train_keras():
    def load_data_keras(hdf_file):
        train = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in loadsets if k
                 not in valset or k not in testset]
        val = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in valset]
        test = [pd.read_hdf(hdf_file, key='p' + str(k)) for k in testset]
        return train, val, test
    print('Keras version: {}'.format(keras_version))

    train_set, val_set, test_set = load_data_keras(file_path)
    all_data = train_set + val_set + test_set

    # normalize data
    s_mean, s_max, s_min = get_norm_stats(all_data)
    all_data_normed = [(a - s_mean) / (s_max - s_min) for a in all_data]

    # print(all_data_normed[0].head(10))

    test_set = all_data_normed.pop()
    val_set = all_data_normed.pop()

    val_in = val_set[Input_param_names].values
    val_out = val_set[Target_param_names].values

    test_in = test_set[Input_param_names].values
    test_out = test_set[Target_param_names].values

    input_dim = test_in.shape[1]

    # create model
    model = Sequential()
    model.add(GRU(32, batch_input_shape=(1, 1, input_dim),
                  consume_less='cpu', stateful=True))
    model.add(Dense(4))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ]

    model.compile(optimizer=sgd, loss='mse')

    """print('in shape: {} out shape: {}'.format(model_in.shape, model_out.shape))
    model.fit(model_in, model_out, batch_size=1, nb_epoch=100,
              validation_data=(val_in, val_out),
              callbacks=callbacks)

    score = model.evaluate(test_in, test_out)
    print(score)"""

    print('Train...')
    for epoch in range(1, 15):
        print('Epoch {}'.format(epoch))
        mean_tr_loss = []
        sample_count = 0
        for sample in all_data_normed:
            sample_count += 1
            print('sample {} of {}'.format(sample_count, len(all_data_normed)))
            # split input and output
            model_in = sample[Input_param_names].values
            model_out = sample[Target_param_names].values

            for i in range(len(sample.index)):
                tr_loss = \
                    model.train_on_batch(
                        np.expand_dims(
                            np.expand_dims(model_in[i, :],
                                           axis=0),
                            axis=0),
                        np.atleast_2d(model_out[i, :]))
                mean_tr_loss.append(tr_loss)
            model.reset_states()

            print('loss training = {}'.format(np.mean(mean_tr_loss)))

        mean_te_loss = []
        for i in range(val_in.shape[0]):
            te_loss = model.test_on_batch(
                np.expand_dims(np.expand_dims(val_in[i, :], axis=0), axis=0),
                np.atleast_2d(val_out[i, :]))
            mean_te_loss.append(te_loss)
        model.reset_states()

        print('loss validation = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')

    print("Test..")

    y_pred = []
    for j in range(test_in.shape[0]):
        batch = test_in[j, :]
        y_pred.append(model.predict_on_batch(batch[np.newaxis, np.newaxis, :]))
    y_pred = np.vstack(y_pred)
    model.reset_states()

    mean_te_loss = []
    for i in range(test_in.shape[0]):
        batch = test_in[i, :]
        batch_y = test_out[i, :]
        te_loss = model.test_on_batch(batch[np.newaxis, np.newaxis, :],
                                      batch_y[np.newaxis, :])
        mean_te_loss.append(te_loss)
    model.reset_states()

    print('loss test = {}'.format(np.mean(mean_te_loss)))
    print('___________________________________')

    time = np.arange(len(test_in.index), dtype=np.float32)
    time /= (2 * 60)
    plt.plot(time, test_out[:, 0])
    plt.plot(time, y_pred[:, 0])


def train_linear():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    print('train')

    #lr_model = LinearRegression()
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
    return pd.DataFrame(preds, columns=y_cols)


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    import pandas as pd
    from tpot import TPOTRegressor
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import LSTM, GRU
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping
    from keras import __version__ as keras_version

    scaler = MinMaxScaler()
    dataset = pd.read_csv(join('input', 'measures.csv'), dtype=np.float32)
    orig_cols = [c for c in dataset.columns if c != profile_id_colname]

    tra_df, val_df, tst_df = munge_data(dataset.copy(), lookback=2)
    tra_df = pd.concat([tra_df, val_df], axis=0)
    x_cols = [col for input in Input_param_names for col in tra_df.columns if
              col[:3] in input]
    y_cols = Target_param_names

    pred_df = pd.DataFrame(np.zeros(tst_df[orig_cols].shape),
                           columns=orig_cols)

    # tpot
    """for target in Target_param_names:
        tpot = TPOTRegressor(verbosity=2, cv=5, random_state=2017,
                             n_jobs=-1, periodic_checkpoint_folder='out')
        tpot.fit(tra_df[x_cols], tra_df[target])
        tpot.export('out/tpotted_{}.py'.format(target))
        pred_df[target] = tpot.predict(tst_df[x_cols])"""

    # linear model
    pred = train_linear()


    actual = \
        dataset[dataset[profile_id_colname].isin(testset)].loc[:, y_cols]
    actual.reset_index(drop=True, inplace=True)

    # untransform prediction
    inversed_pred = pd.DataFrame(
        scaler.inverse_transform(pred_df),
        columns=orig_cols).loc[:, y_cols]

    print('mse: {} K²'.format(mean_squared_error(actual,
                                                 inversed_pred)))

    # plots
    """plt.plot(actual)
    plt.plot(inversed_pred)
    plt.show()"""
