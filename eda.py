from os.path import join
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from kirgsn.data import DataManager

if __name__ == '__main__':
    dm = DataManager(join('input', 'measures.csv'))
    tra_df, val_df, tst_df = dm.get_featurized_sets()
    print(len(dm.cl.x_cols))
    for i, x_col in enumerate(dm.cl.x_cols):
        plt.subplot(6,7,i+1)
        plt.scatter(tra_df[x_col], tra_df[dm.cl.y_cols[0]])
        plt.xlabel(x_col)
    plt.show()
