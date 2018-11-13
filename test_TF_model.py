# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from libact.models.mymodel_keras import MyModel
from libact.models.mymodel_TF import MyModelTF
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)


def main():
    # n_labeled = 10      # number of samples that are initially labeled

    # Load dataset
    # val_data
    # X_test = np.load('../ziqi/train_data_5000.npy')
    # y_test_csv = pd.read_csv('../ziqi/train_data_label_5000.csv')
    # y_test = np.array([i for i in y_test_csv['label']])
    # tst_ds = Dataset(X_test, y_test)
    #
    # noisy_data
    y_csv = pd.read_csv('data_label.csv')
    y_test = y_csv['label']
    y_test = [i for i in y_test]
    y_test = np.array(y_test)
    
    X_pool = np.load('data.npy')
    y_pool = [None] * len(X_pool)
    pool_ds = Dataset(X_pool, y_pool)


    # Comparing UncertaintySampling strategy with entropy uncertainty sampling
    model_folder = './model/vgg-x30-9982.pb'
    
    # model predict 
    print('model predict....')
    model = MyModelTF(model_folder, batch_size=256)
    score = model.predict(X_pool)
    score = pd.DataFrame(score)
    score.to_csv('noisy_predict_all.csv', index=False)
    # score = np.argmax(score, axis=1)

    # all_csv = {'indexs':y_test_csv['filename'], 'pre_label': score, 'name_label': y_test}
    # all_csv = pd.DataFrame(all_csv)
    # all_csv.to_csv('../ziqi/result/train_predict_result.csv')
    # acc_test = model.score(tst_ds)
    # print('acc: ', acc_test)

    print('make query')
    qs = UncertaintySampling(pool_ds, method='entropy', model=MyModelTF(model_folder, batch_size=128))
    ask_ids, labels = qs.make_query(return_label=True, n_instances=5000)
    print('ask_ids : ', np.shape(ask_ids))
    
    csv = {'indexs':ask_ids, 'pre_label': labels, 'name_label': y_test[ask_ids]}
    csv = pd.DataFrame(csv)
    csv.to_csv('./result/sample_result.csv', index=False)


if __name__ == '__main__':
    main()
