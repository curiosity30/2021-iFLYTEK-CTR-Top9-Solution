# Load Data
import pandas as pd
import numpy as np
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
path = 'C:/Users/Administrator/Python_Learning/Competition/iFLYTEK-CTR/data/'
train_csv = 'train.csv'
test_csv = 'test.csv'
sample_submit = 'sample_submit.csv'
id_col = 'id'
target = 'isClick'
category_cols = ['user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_id', 'gender', 'user_group_id']
count_cols = ['user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_id', 'gender', 'user_group_id',
              'age_level', 'user_depth', 'var_1']
target_encode_cols = ['user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_id', 'user_group_id',
                      'gender', 'age_level', 'user_depth', 'var_1']
label_cols = []
cross_feature = []


# =================================== reduce memory function =============================================
def reduce_mem_usage(df, verbose=True):
    '''自定义用来降低内存空间的函数'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# =================================== preprocess data =============================================
def read_data():
    print('reading data...')
    df_train = pd.read_csv(path + train_csv)
    df_test = pd.read_csv(path + test_csv)
    df_submit = pd.read_csv(path + sample_submit)
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    print('train data shape: ', df_train.shape)
    print('test  data shape: ', df_test.shape)
    return df_train, df_test, df_submit


def gender_map(x):
    if x == 'Female': sex = 0
    elif x == 'Male': sex = 1
    else: sex = -1
    return sex


def user_group_map(x):
    age_level = -1
    if x == -1: age_level = -1
    elif x == 0: age_level = 0
    elif x == 1: age_level = 1
    elif x == 7: age_level = 2
    elif x == 2: age_level = 3
    elif x == 8: age_level = 4
    elif x == 3: age_level = 5
    elif x == 9: age_level = 6
    elif x == 4: age_level = 7
    elif x == 10: age_level = 8
    elif x == 5: age_level = 9
    elif x == 11: age_level = 10
    elif x == 6: age_level = 11
    elif x == 12: age_level = 12
    return age_level


def preprocess_data(df):
    df['gender'] = df['gender'].apply(gender_map)
    df['user_group_id'] = df['user_group_id'].fillna(-1)
    df['age_level'] = df['age_level'].fillna(-1)
    df['user_depth'] = df['user_depth'].fillna(-1)
    df['user_group_id'] = df['user_group_id'].apply(user_group_map)
    return df


def product_embedding(df):
    user_lst = df['user_id'].unique().tolist()
    prod_sentences = []
    emb_matrix_mean = []
    emb_matrix_max = []
    emb_size = 5

    for user in tqdm(user_lst):
        temp_seq = df[df['user_id'] == user]['product'].tolist()
        prod_sentences.append(temp_seq)

    model = Word2Vec(sentences=prod_sentences, vector_size=emb_size, min_count=1, window=1, sg=1, hs=0, seed=42)
    model.build_vocab(prod_sentences)
    model.train(prod_sentences, total_examples=model.corpus_count, epochs=model.epochs)

    for seq in tqdm(prod_sentences):
        vec = []
        for w in seq:
            vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix_mean.append(np.mean(vec, axis=0))
            emb_matrix_max.append(np.max(vec, axis=0))
        else:
            emb_matrix_mean.append([0] * emb_size)
            emb_matrix_max.append([0] * emb_size)

    emb_matrix_mean = pd.DataFrame(emb_matrix_mean)
    emb_matrix_mean.columns = [f'product_embed_{i}_mean' for i in range(emb_size)]
    emb_matrix_mean['user_id'] = user_lst

    emb_matrix_max = pd.DataFrame(emb_matrix_max)
    emb_matrix_max.columns = [f'product_embed_{i}_max' for i in range(emb_size)]
    emb_matrix_max['user_id'] = user_lst

    df = pd.merge(df, emb_matrix_mean, how='left', on='user_id')
    df = pd.merge(df, emb_matrix_max, how='left', on='user_id')
    return df


# =================================== Feature Engineer =============================================
def time_bin_7(x):
    if x in [23, 0, 1]:
        return 0
    elif x in [2, 3, 4]:
        return 1
    elif x in [5, 6, 7]:
        return 2
    elif x in [8, 9, 10, 11]:
        return 3
    elif x in [12, 13]:
        return 4
    elif x in [14, 15, 16, 17]:
        return 5
    elif x in [18, 19, 20, 21, 22]:
        return 6


def time_bin_4(x):
    if 0 <= x <= 6:
        return 1
    elif 7 <= x <= 12:
        return 2
    elif 13 <= x <= 17:
        return 3
    elif 18 <= x <= 23:
        return 4


def get_time_fea(df):
    df['day'] = df['date'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    df['hour'] = df['date'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    df['minute'] = df['date'].apply(lambda x: int(x.split(' ')[-1].split(':')[1]))
    df['time_id'] = (df['day'] - 2) * 24 + df['hour']
    df['isHourStart'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 10 else 0)
    df['isHourEnd'] = df['hour'].apply(lambda x: 1 if 50 <= x <= 59 else 0)
    df['isHourMid'] = df['hour'].apply(lambda x: 1 if 25 <= x <= 35 else 0)
    df['time_bin_7'] = df['hour'].apply(time_bin_7)
    # df['time_bin_4'] = df['hour'].apply(time_bin_4)
    count_cols.extend(['day', 'hour', 'time_bin_7'])
    return df


def cross_self_2_fea(df, cross_cols):
    for i in range(len(cross_cols)):
        for j in range(i+1, len(cross_cols)):
            col = cross_cols[i] + '_' + cross_cols[j]
            df[col] = df[cross_cols[i]].astype(str) + '_' + df[cross_cols[j]].astype(str)
            cross_feature.append(col)
            label_cols.append(col)
            count_cols.append(col)
    return df


def cross_self_3_fea(df, cross_cols):
    for i in range(len(cross_cols)):
        for j in range(i+1, len(cross_cols)):
            for k in range(j+1, len(cross_cols)):
                col = cross_cols[i] + '_' + cross_cols[j]
                df[col] = df[cross_cols[i]].astype(str) + '_' + df[cross_cols[j]].astype(str)
                cross_feature.append(col)
                label_cols.append(col)
                count_cols.append(col)
    return df


def cross_2_fea(df, first_cols, second_cols):
    for fea1 in first_cols:
        for fea2 in second_cols:
            if fea1 != fea2:
                col = fea1 + '_' + fea2
                df[col] = df[fea1].astype(str) + '_' + df[fea1].astype(str)
                cross_feature.append(col)
                label_cols.append(col)
                count_cols.append(col)
    return df


def make_cross_feature(df):
    item_cols = ['product', 'campaign_id', 'webpage_id']
    user_cols = ['gender', 'age_level', 'user_depth']

    df = cross_self_2_fea(df, cross_cols=item_cols)
    df = cross_self_2_fea(df, cross_cols=user_cols)
    df = cross_2_fea(df, first_cols=item_cols, second_cols=user_cols)

    df['depth_var'] = df['user_depth'].astype(str) + '_' + df['var_1'].astype(str)
    df['age_var'] = df['age_level'].astype(str) + '_' + df['var_1'].astype(str)
    df['gender_age_depth_var'] = df['user_group_id'].astype(str) + '_' + df['user_depth'].astype(str) + '_' + df['var_1'].astype(str)
    df['product_campaign_webpage_id'] = df['product'].astype(str) + '_' + df['campaign_id'].astype(str) + '_' + df['webpage_id'].astype(str)
    # df['gender_age_depth'] = df['gender'].astype(str) + '_' + df['age_level'].astype(str) + '_' + df['user_depth'].astype(str)

    label_cols.extend(['product_campaign_webpage_id', 'gender_age_depth_var', 'depth_var', 'age_var'])
    count_cols.extend(['product_campaign_webpage_id', 'gender_age_depth_var', 'depth_var', 'age_var'])
    cross_feature.extend(['product_campaign_webpage_id', 'gender_age_depth_var', 'depth_var', 'age_var'])
    target_encode_cols.extend(['product_campaign_webpage_id', 'gender_age_depth_var'])
    # target_encode_cols.extend(['campaign_id_age_level', 'webpage_id_gender', 'product_user_depth'])
    return df


def cross_category_fea(df):
    df['user_item_count'] = df.groupby('user_id')['product'].transform('count').values
    df['user_item_nunique'] = df.groupby('user_id')['product'].transform('nunique').values
    df['user_category_count'] = df.groupby('user_id')['product_category_id'].transform('count').values
    df['user_category_nunique'] = df.groupby('user_id')['product_category_id'].transform('nunique').values
    df['item_user_count'] = df.groupby('product')['user_id'].transform('count').values
    df['item_user_nunique'] = df.groupby('product')['user_id'].transform('nunique').values

    df['user_campaign_nunique'] = df.groupby('user_id')['campaign_id'].transform('nunique').values
    df['user_webpage_nunique'] = df.groupby('user_id')['campaign_id'].transform('nunique').values
    df['product_campaign_nunique'] = df.groupby('product')['campaign_id'].transform('nunique').values
    df['product_webpage_nunique'] = df.groupby('product')['webpage_id'].transform('nunique').values
    df['campaign_product_nunique'] = df.groupby('campaign_id')['product'].transform('nunique').values
    df['campaign_webpage_nunique'] = df.groupby('campaign_id')['webpage'].transform('nunique').values
    # df['webpage_product_nunique'] = df.groupby('webpage_id')['product'].transform('nunique').values
    # df['webpage_campaign_nunique'] = df.groupby('webpage_id')['campaign_id'].transform('nunique').values
    return df


def stat_fea(df, stat_label, cate_cols):
    for col in cate_cols:
        df[f'{stat_label}_by_{col}_mean'] = df.groupby(col)[stat_label].transform('mean').values
        df[f'{stat_label}_by_{col}_std'] = df.groupby(col)[stat_label].transform('std').values
    return df


def cross_stat_fea(df, stat_label, cate_cols):
    for col in cate_cols:
        df[f'day_{stat_label}_{col}_mean'] = df.groupby(['day', col])[stat_label].transform('mean')
    return df


def time_stat_fea(df):
    stat_cols = ['user_id', 'product', 'campaign_id', 'webpage_id']
    df['user_day_count'] = df.groupby(['day', 'user_id'])['user_id'].transform('count').values
    df['user_time_bin_count'] = df.groupby(['time_bin_7', 'user_id'])['user_id'].transform('count').values
    for stat_label in stat_cols:
        df[f'time_id_{stat_label}_count'] = df.groupby('time_id')[stat_label].transform('count').values
    return df


def one_hot_stat_fea_of_user(df):
    """ user 对于 'product', 'product_category_id', 'campaign_id' 等的偏好比率 """
    # one_hot_cols = ['product', 'product_category_id', 'campaign_id', 'webpage_id', 'time_bin_7']
    one_hot_cols = ['product', 'product_category_id', 'campaign_id', 'webpage_id', 'product_campaign_webpage_id', 'time_bin_7']
    for fea in one_hot_cols:
        dummy_df = pd.get_dummies(df[fea], prefix=fea)
        dummy_cols = dummy_df.columns
        df = pd.concat([df, dummy_df], axis=1)
        for col in dummy_cols:
            df[f'user_{col}_ratio'] = df.groupby('user_id')[col].transform('mean')
        df = df.drop(dummy_cols, axis=1)
    return df


def one_hot_stat_fea_of_product(df):
    """ product 对于 'campaign_id', 'webpage_id' 的偏好比率 """
    one_hot_cols = ['campaign_id', 'webpage_id']
    for fea in one_hot_cols:
        dummy_df = pd.get_dummies(df[fea], prefix=fea)
        dummy_cols = dummy_df.columns
        df = pd.concat([df, dummy_df], axis=1)
        for col in dummy_cols:
            df[f'product_{col}_ratio'] = df.groupby('product')[col].transform('mean')
        df = df.drop(dummy_cols, axis=1)
    return df


def one_hot_stat_fea_of_context(df):
    """ campaign/webpage 对于 'gender', 'age_level' 的 吸引比率"""
    # one_hot_cols = ['gender', 'age_level']
    one_hot_cols = ['gender', 'user_group_id', 'var_1']
    for fea in one_hot_cols:
        dummy_df = pd.get_dummies(df[fea], prefix=fea)
        dummy_cols = dummy_df.columns
        df = pd.concat([df, dummy_df], axis=1)
        for col in dummy_cols:
            df[f'campaign_{col}_ratio'] = df.groupby('campaign_id')[col].transform('mean')
            df[f'webpage_{col}_ratio'] = df.groupby('webpage_id')[col].transform('mean')
        df = df.drop(dummy_cols, axis=1)
    return df


def _his_click_rate(df, f1, window_size=2):
    fea_name = '{}_his_{}_click_rate'.format(f1, window_size)
    df[fea_name] = 0
    for i in tqdm(range(3, 8)):
        df_t = df.loc[((df['day'] >= i - window_size) & (df['day'] < i))]
        idxs = df['day'] == i
        df.loc[idxs, fea_name] = df.loc[idxs, f1].map(df_t.groupby(f1)['isClick'].mean())
    return df


def count_encoding(df):
    for col in tqdm(count_cols):
        df[f'{col}_count'] = df[col].map(df[col].value_counts())
    return df


def label_encoding(df):
    for col in tqdm(label_cols):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype('int')
    return df


# =================================== target encoding =============================================
def stat(df, df_merge, group_by, agg):
    group = df.groupby(group_by).agg(agg)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()

    return df_merge


def statis_feat(df_know, df_unknow):
    """只需要修改target_encode_cols"""
    for f in tqdm(target_encode_cols):
        df_unknow = stat(df_know, df_unknow, [f], {target: ['mean', 'std']})
    return df_unknow


def target_encoding(df):
    train = df[~df[target].isnull()]
    train = train.reset_index(drop=True)
    test = df[df[target].isnull()]

    df_stas_feat = None
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    print('target encoding...')

    for tra_index, val_index in kfold.split(train, train[target]):
        df_fold_train = train.iloc[tra_index]
        df_fold_val = train.iloc[val_index]

        df_fold_val = statis_feat(df_fold_train, df_fold_val)
        df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

        del (df_fold_train)
        del (df_fold_val)
        gc.collect()

    test = statis_feat(train, test)
    df = pd.concat([df_stas_feat, test], axis=0)

    del (df_stas_feat)
    del (train)
    del (test)
    gc.collect()

    return df


# =================================== process data =============================================
def process_data(df):
    df = get_time_fea(df)
    df = _his_click_rate(df=df, f1='user_id', window_size=2)
    df = make_cross_feature(df)
    df = one_hot_stat_fea_of_user(df)
    df = one_hot_stat_fea_of_context(df)
    df = time_stat_fea(df)
    df = stat_fea(df, stat_label='user_depth', cate_cols=category_cols)
    df = stat_fea(df, stat_label='user_depth', cate_cols=cross_feature)
    df = stat_fea(df, stat_label='user_group_id', cate_cols=category_cols)
    df = stat_fea(df, stat_label='var_1', cate_cols=category_cols)
    # df = cross_stat_fea(df, stat_label='user_depth', cate_cols=category_cols)
    df = count_encoding(df)
    df = label_encoding(df)
    df = target_encoding(df)
    return df


# =================================== LightGBM model =============================================
def cv_model(clf, train_x, train_y, test_x, clf_name='lgb'):
    folds = 5
    seed = 2021
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train_pred_lst = np.zeros(train_x.shape[0])
    test_pred_lst = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kfold.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y = train_x.iloc[train_index], train_y.iloc[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y.iloc[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'min_child_weight': 5,
            'num_leaves': 2 ** 7,
            'lambda_l2': 10,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.9,
            'bagging_freq': 4,
            'learning_rate': 0.01,
            'seed': 2021,
            'n_jobs': -1,
            'silent': True,
            'verbose': -1,
        }

        model = clf.train(params, train_matrix, 10000,
                          valid_sets=[train_matrix, valid_matrix],
                          verbose_eval=200,
                          early_stopping_rounds=100)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        train_pred_lst[valid_index] = val_pred
        test_pred_lst += test_pred / kfold.n_splits
        cv_scores.append(round(roc_auc_score(val_y, val_pred), 5))

        print(cv_scores)

    mean_auc = round(np.mean(cv_scores), 5)
    std_auc = round(np.std(cv_scores), 4)
    print('AUC list: ', cv_scores)
    print('AUC mean: ', mean_auc)
    print('AUC std:  ', std_auc)
    return train_pred_lst, test_pred_lst, mean_auc


if __name__ == '__main__':
    # read data
    train, test, submit = read_data()
    df_feature = pd.concat([train, test], axis=0)

    # feature engineer
    df_feature = preprocess_data(df_feature)
    df_feature = product_embedding(df_feature)
    df_feature = process_data(df_feature)

    # train test split
    train = df_feature[df_feature[target].notnull()]
    test = df_feature[df_feature[target].isnull()]

    # drop useless features
    useless_cols = [id_col, target, 'date', 'time_bin_4']
    all_cols = [col for col in train.columns if col not in useless_cols]
    x_train = train[all_cols]
    x_test = test[all_cols]
    y_train = train[target]
    print('x_train shape: ', x_train.shape)
    print('x_test shape: ', x_test.shape)
    print('category cols: ', category_cols)
    print('count cols: ', count_cols)
    print('label cols: ', label_cols)
    print('cross cols: ', cross_feature)

    # lightGBM model training
    lgb_train, lgb_test, lgb_score = cv_model(lgb, x_train, y_train, x_test, clf_name='lgb')

    # submission
    submit_path = 'C:/Users/Administrator/Python_Learning/Competition/iFLYTEK-CTR/submission/'
    submit[target] = lgb_test
    submit.to_csv(submit_path + f'submit_{lgb_score}.csv', index=False)
