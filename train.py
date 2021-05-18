import pandas as pd
import tensorflow as tf
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from model import xDeepFM_MTL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ONLINE_FLAG = False
loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

if __name__ == "__main__":
    epochs=5
    batch_size=512

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
    dense_features = ['video_duration']  # 'creat_time',
    target = ['finish', 'like']

    try:
        # read data from pkl directly
        data=pd.read_pickle('data_icme19.pkl')
        train_size = int(data.shape[0] * (1 - VALIDATION_FRAC))
        print('read_pickle ok')
    except:    
        # data = pd.read_csv('./input/final_track2_train_200.txt', sep='\t',
        data = pd.read_csv('./input/final_track2_train.txt', sep='\t',
                           names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like',
                                  'music_id', 'did', 'creat_time', 'video_duration'])
        if ONLINE_FLAG:
            test_data = pd.read_csv('./input/final_track2_test_no_anwser.txt', sep='\t',
                                    names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
                                           'like', 'music_id', 'did', 'creat_time', 'video_duration'])
            train_size = data.shape[0]
            data = data.append(test_data)
        else:
            train_size = int(data.shape[0] * (1 - VALIDATION_FRAC))


        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )


        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])
        
        data.to_pickle('data_icme19.pkl')
        print('to_pickle ok')

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())  for feat in sparse_features]+[DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL(linear_feature_columns, dnn_feature_columns,dnn_hidden_units=(128, 128))
    # model = xDeepFM_MTL({"sparse": sparse_feature_list,
    #                      "dense": dense_feature_list})

    def auc(y_true, y_pred):
         auc = tf.metrics.auc(y_true, y_pred)[1]
         tf.keras.backend.get_session().run(tf.local_variables_initializer())
         return auc

    model.compile("adagrad", "binary_crossentropy", metrics=['binary_crossentropy',auc], loss_weights=loss_weights, )

    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=batch_size, epochs=epochs, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2 ** 14)

    else:
        history = model.fit(train_model_input, train_labels,
                            batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_model_input, test_labels))

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
