from os.path import join
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.data import DataManager
from preprocessing import config as cfg

if __name__ == '__main__':

    dm = DataManager(cfg.data_cfg['file_path'])
    tra_df, val_df, tst_df = dm.get_featurized_sets()
    print('X_Cols: n Features:', len(dm.cl.x_cols))
    df_to_plot = tra_df.sample(500)
    for col in df_to_plot:
        print(col)
    #g = sns.pairplot(df_to_plot, kind="reg")
    #g = g.map_offdiag(sns.jointplot)
    """for i, x_col in enumerate(dm.cl.x_cols):
        plt.subplot(6, 7, i+1)

        plt.scatter(tra_df[x_col], tra_df[dm.cl.y_cols[0]])
        #sns.jointplot(x=x_col, y=dm.cl.y_cols[0], data=df_to_plot, kind='hex')
        plt.xlabel(x_col)
        #tra_df.plot.hexbin(x=x_col, y=dm.cl.y_cols[0], gridsize=20)"""
    #sns.plt.show()
