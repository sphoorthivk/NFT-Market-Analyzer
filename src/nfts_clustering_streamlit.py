import streamlit as st
from turtle import title

from utils.visualization import color_palette_mapping
from utils.numerical import nan_average
import os
import warnings
import sqlite3

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

import numpy as np
import nfts.dataset

import pandas as pd
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer
from kneed import KneeLocator
import pyclustertend

### --- Streamlit Configuration --- ###

st.set_page_config(title="nfts Clustering", )

nfts_merged_df = pd.read_csv("data/nfts_merged.csv")

### --- Preprocessing: Imputation of NaN Values --- ###

# MCM (Most Common Value) Imputation for mints_values, transfers_values, market_values
nfts_merged_df.mints_avg_transaction_value.fillna(nan_average(nfts_merged_df, "mints_avg_transaction_value"), inplace=True)
nfts_merged_df.transfers_avg_transaction_value.fillna(nan_average(nfts_merged_df, "transfers_avg_transaction_value"), inplace=True)
nfts_merged_df.avg_market_value.fillna(nan_average(nfts_merged_df, "avg_market_value"), inplace=True)

# 0 for the other values (since they represents counts)
nfts_merged_df.mints_timestamp_range.fillna(0, inplace=True)
nfts_merged_df.transfers_count.fillna(0, inplace=True)
nfts_merged_df.num_owners.fillna(0, inplace=True)


### --- Principal Component Analysis --- ###

# Selecting significative features
features = ["mints_avg_transaction_value", "mints_timestamp_range", "transfers_avg_transaction_value",
             "transfers_count", "num_owners", "avg_market_value"]

# Separating out the features
x_pca = nfts_merged_df.loc[:, features]

# Standardizing the features
x_pca = StandardScaler().fit_transform(x_pca)
standardized_df = pd.DataFrame(x_pca, columns = features)

# pca with all components (6)
pca = PCA()

principal_components = pca.fit_transform(x_pca)
print("Variance explained by each Principal Component:", pca.explained_variance_ratio_)
cumulative_sum_variance = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative sum of Variance explained by each Principal Component:", cumulative_sum_variance)
print("Variance explained by the 4 Principal Components alone:", sum(pca.explained_variance_ratio_[0:4]))

fig = plt.figure(figsize=(10,5))
plt.plot(range(1, len(cumulative_sum_variance)+1), cumulative_sum_variance)
plt.xlabel("Number of Components")
_ = plt.ylabel("Explained Variance (%)")
_ = plt.title("Variance Accumulation")

st.pyplot(fig)