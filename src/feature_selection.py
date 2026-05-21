import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42

def ovr_select_features_mi(X, y, target_class, n_top):
    y_bin = (y == target_class).astype(int)
    mi    = mutual_info_classif(X, y_bin, random_state=SEED)
    idx   = np.argsort(mi)[::-1][:n_top]
    return sorted(idx.tolist())

def ovr_select_features_xgb(X, y, target_class, n_top):
    y_bin = (y == target_class).astype(int)
    clf   = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           objective='binary:logistic', tree_method='hist',
                           random_state=SEED, n_jobs=-1, eval_metric='logloss')
    clf.fit(X, y_bin, sample_weight=compute_sample_weight('balanced', y_bin))
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:n_top]
    return sorted(idx.tolist())

def ovr_select_features_rfe(X, y, target_class, n_top):
    y_bin = (y == target_class).astype(int)
    svc   = LinearSVC(C=1.0, max_iter=5000, random_state=SEED, class_weight='balanced')
    rfe   = RFE(estimator=svc, n_features_to_select=n_top, step=3)
    rfe.fit(X, y_bin)
    return sorted(np.where(rfe.support_)[0].tolist())