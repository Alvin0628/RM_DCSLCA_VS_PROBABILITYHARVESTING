import numpy as np
from sklearn.neighbors import NearestNeighbors

def dcs_lca_predict(pool, X_dsel, y_dsel, X_test, k=7):
    X_dsel_arr = np.array(X_dsel)
    X_test_arr = np.array(X_test)
    y_dsel_arr = np.array(y_dsel)

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_dsel_arr)
    _, nbr_idx = nn.kneighbors(X_test_arr)

    final_preds = []
    for i, neighbors in enumerate(nbr_idx):
        y_nbr = y_dsel_arr[neighbors]
        X_nbr = X_dsel_arr[neighbors]
        x_i   = X_test_arr[i:i+1]
        best_lca, best_pred = -1.0, None
        for clf in pool:
            pred_class = clf.predict(x_i)[0]
            mask = y_nbr == pred_class
            lca  = np.mean(clf.predict(X_nbr[mask]) == y_nbr[mask]) if mask.sum() > 0 else 0.0
            if lca > best_lca:
                best_lca, best_pred = lca, pred_class
        final_preds.append(best_pred)
    return np.array(final_preds)