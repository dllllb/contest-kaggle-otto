import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing.label import label_binarize

import ds_tools.dstools.ml.xgboost_tools as xgb


def submission(est):
    df = pd.read_csv('train.csv.gz', index_col='id')

    features = df.drop(['target'], axis=1)
    labels = df.target.apply(lambda e: e[6:]).astype(np.int16)-1
    
    model = est.fit(features, labels)

    df_test = pd.read_csv('test.csv.gz', index_col='id')

    y_pred = model.predict_proba(df_test)

    res_df = pd.DataFrame(y_pred, columns=['Class_%d' % n for n in range(1, 10)], index=df_test.index)
    res_df.to_csv('results.csv', index_label='id')


def otto_params(overrides):
    defaults = {
        'est': 'xgb',
        'n_folds': 3,
        "eta": 0.1,
        "num_rounds": 10000,
        "max_depth": 10,
        "min_child_weight": 4,
        "gamma": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        'calibration': False,
        'calibration_folds': 5,
        'calibration_method': 'isotonic',
        'epochs': 3,
        'dropout': .3,
        'batch_size': 256,
        'lr': .01,
        'decay': .001,
    }
    
    return {**defaults, **overrides}
    

def xgb_est(params):
    keys = {
        'eta',
        'num_rounds',
        'max_depth',
        'min_child_weight',
        'gamma',
        'subsample',
        'colsample_bytree'
    }
    
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 9,
        "scale_pos_weight": 1,
        "verbose": 10,
        "eval_metric": "mlogloss",
        **{k: v for k, v in params.items() if k in keys}
    }
    
    return xgb.XGBoostClassifier(**xgb_params)


def keras_est(params):
    from keras import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from keras.wrappers.scikit_learn import KerasClassifier

    def create_model():
        dropout = params['dropout']
        model = Sequential()
        model.add(Dense(128, input_dim=93, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(9, activation='softmax'))

        optimizer = Adam(lr=params['lr'], decay=params['decay'])
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    est = KerasClassifier(build_fn=create_model, epochs=params['epochs'], batch_size=params['batch_size'])
    return est


def otto_dataset(params):
    df = pd.read_csv('train.csv.gz', index_col='id', nrows=params.get('n_rows'))

    features = df.drop(['target'], axis=1)
    labels = df.target.apply(lambda e: e[6:]).astype(np.int16) - 1
    if params['est'] == 'keras':
        labels = label_binarize(labels, classes=sorted(set(labels)))

    return features, labels


def otto_estimator(params):
    est_type = params['est']
    if est_type == 'xgb':
        est = xgb_est(params)
    elif est_type == 'keras':
        est = keras_est(params)
    else:
        raise AttributeError(f'unknown estimator type: {est_type}')
    
    if params['calibration']:
        est = CalibratedClassifierCV(
            base_estimator=est,
            cv=params['calibration_folds'],
            method=params['calibration_method'])
    
    return est


def otto_experiment(overrides):
    params = otto_params(overrides)

    results = run_experiment(
        params=params,
        est=otto_estimator,
        dataset=otto_dataset,
        scorer='neg_log_loss')

    update_model_stats('results.json', params, results)


def test_otto_experiment_xgb():
    overrides = {
        'n_folds': 2,
        'num_rounds': 5
    }

    params = otto_params(overrides)

    results = run_experiment(
        params=params,
        est=otto_estimator,
        dataset=otto_dataset,
        scorer='neg_log_loss')

    print(results)


def test_otto_experiment_keras():
    overrides = {
        'est': 'keras',
        'n_folds': 2,
        'num_rounds': 5,
        'epochs': 1,
    }

    params = otto_params(overrides)

    results = run_experiment(
        params=params,
        est=otto_estimator,
        dataset=otto_dataset,
        scorer='neg_log_loss')

    print(results)


def update_model_stats(stats_file, params, results):
    import json
    import os.path

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []

    stats.append({**results, **params})

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)


def run_experiment(est, dataset, scorer, params):
    import time

    start = time.time()
    cv = params['n_folds']
    features, target = dataset(params)
    scores = cv_test(est(params), features, target, scorer, cv)
    exec_time = time.time() - start
    return {**scores, 'exec-time-sec': exec_time}


def cv_test(est, features, target, scorer, cv):
    scores = cross_val_score(est, features, target, scoring=scorer, cv=cv)
    return {'score-mean': scores.mean(), 'score-std': scores.std()}
