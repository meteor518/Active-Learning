import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from libact.models.mymodel_keras import MyModel

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)


def main():
    # n_labeled = 10      # number of samples that are initially labeled

    # Load dataset
    # val data
    X_test = pd.read_pickle('image_val.pkl')
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    X_test = X_test / 255.0
    y_test = pd.read_csv('data_clean_label_val_lianxing.csv')
    y_test = np.array([i for i in y_test['lianxing']])
    tst_ds = Dataset(X_test, y_test)

    #
    X_pool = np.load('./data/unlabel.npy')
    X_pool = X_pool / 255.
    X_pool = np.transpose(X_pool, (0, 3, 1, 2))
    y_pool = [None] * len(X_pool)
    pool_ds = Dataset(X_pool, y_pool)


    # Comparing UncertaintySampling strategy with RandomSampling.
    model_folder = './model/initial.h5'
    qs = UncertaintySampling(pool_ds, method='lc', model=MyModel(model_folder, input_shape=(3, 224, 224), nclasses=3))
    model = MyModel(model_folder, input_shape=(3, 224, 224), nclasses=3)

    acc_test = model.score(tst_ds)
    print('acc: ', acc_test)

    print('make query')
    ask_ids= qs.make_query(n_instances=100)
    print('ask_ids : ', np.shape(ask_ids), ask_ids[:2])


if __name__ == '__main__':
    main()
