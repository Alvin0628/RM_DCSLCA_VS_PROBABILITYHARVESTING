import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

def clean_knn_overlap(df, cat_cols, k_clean=10, purity_th=0.40):
    te_clean   = TargetEncoder(cols=cat_cols, smoothing=10)
    X_full_raw = df.drop(columns=['Target'])
    y_full     = df['Target']
    X_enc_full = te_clean.fit_transform(X_full_raw, y_full)
    sc_clean   = StandardScaler()
    X_sc_full  = sc_clean.fit_transform(X_enc_full)
    y_full_arr = y_full.values

    nn = NearestNeighbors(n_neighbors=k_clean + 1, n_jobs=-1)
    nn.fit(X_sc_full)
    nbr_idx = nn.kneighbors(X_sc_full, return_distance=False)[:, 1:]
    purity  = np.array([np.mean(y_full_arr[idx] == y_full_arr[i]) for i, idx in enumerate(nbr_idx)])
    clean_mask_full = purity >= purity_th
    
    return df[clean_mask_full].copy(), clean_mask_full, X_sc_full, y_full_arr