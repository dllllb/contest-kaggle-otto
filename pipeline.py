import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import numpy as np

import ds_tools.dstools.ml.xgboost_tools as xgb


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

        
def run_experiment(evaluator, params, stats_file):    
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})


def cv_test(est, n_folds, n_rows):
    df = pd.read_csv('train.csv.gz', index_col='id', nrows=n_rows)

    features = df.drop(['target'], axis=1)
    labels = df.target.apply(lambda e: e[6:]).astype(np.int16)-1
    
    scores = cross_val_score(estimator=est, X=features, y=labels, cv=n_folds, scoring='neg_log_loss')
    return {'lloss-mean': scores.mean(), 'llos-std': scores.std()}
    
    
def submission(est):
    df = pd.read_csv('train.csv.gz', index_col='id')

    features = df.drop(['target'], axis=1)
    labels = df.target.apply(lambda e: e[6:]).astype(np.int16)-1
    
    model = est.fit(features, labels)

    df_test = read_csv('test.csv.gz', index_col='id')

    y_pred = model.predict_proba(df_test)

    res_df = DataFrame(y_pred, columns=['Class_%d' % n for n in range(1, 10)], index=df_test.index)
    res_df.to_csv('results.csv', index_label='id')

    
def init_params(overrides):
    defaults = {
        'valid_mode': 'cv',
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
    }
    
    return {**defaults, **overrides}
    

def init_xgb_est(params):
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


def validate(params):
    est = init_xgb_est(params)
    
    if params['calibration']:
        est = CalibratedClassifierCV(
            base_estimator=est,
            cv=params['calibration_folds'],
            method=params['calibration_method'])
    
    return cv_test(est, params['n_folds'], params.get('n_rows')) 


def test_validate():
    params = {
        'n_folds': 2,
        #'calibration': True,
        'num_rounds': 5
    }
    
    print(validate(params))
