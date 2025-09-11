from turtle import title
import streamlit as st

from utils.visualization import color_palette_mapping
from utils.numerical import nan_average
# import os
# import sqlite3
import pickle
import io 

import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits import mplot3d

import seaborn as sns
import plotly.express as px

import numpy as np
# import nfts.dataset

import pandas as pd
# import math
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer
from kneed import KneeLocator
# import pyclustertend

### --- Streamlit Configuration --- ###

st.set_page_config(page_title="nfts Clutering", page_icon="img/NFT.png", layout="wide")
style.use("seaborn")

nfts_merged_df = pd.read_csv("data/nfts_merged.csv")
pickle_dir = "data/pickle_vars/"

### --- Preprocessing: Imputation of NaN Values --- ###
# TODO: save df already filled
# MCM (Most Common Value) Imputation for mints_values, transfers_values, market_values
nfts_merged_df.mints_avg_transaction_value.fillna(nan_average(nfts_merged_df, "mints_avg_transaction_value"), inplace=True)
nfts_merged_df.transfers_avg_transaction_value.fillna(nan_average(nfts_merged_df, "transfers_avg_transaction_value"), inplace=True)
nfts_merged_df.avg_market_value.fillna(nan_average(nfts_merged_df, "avg_market_value"), inplace=True)

# 0 for the other values (since they represents counts)
nfts_merged_df.mints_timestamp_range.fillna(0, inplace=True)
nfts_merged_df.transfers_count.fillna(0, inplace=True)
nfts_merged_df.num_owners.fillna(0, inplace=True)

custom_cols = [nfts_merged_df['name'], nfts_merged_df['symbol'], 
                round(nfts_merged_df['avg_market_value']*100)/100, nfts_merged_df['num_owners']]
custom_template = """<b>%{customdata[0]} (%{customdata[1]})</b><br><br>
Avg Value: %{customdata[2]} eth<br>
Owners: %{customdata[3]}
<extra></extra>"""

### --- Principal Component Analysis --- ###
st.header("Principal Component Analysis")

# Selecting significative features
features = ["mints_avg_transaction_value", "mints_timestamp_range", "transfers_avg_transaction_value",
             "transfers_count", "num_owners", "avg_market_value"]

# Separating out the features
x_pca = nfts_merged_df.loc[:, features]

# Standardizing the features
x_pca = StandardScaler().fit_transform(x_pca)
standardized_df = pd.DataFrame(x_pca, columns = features)

## -- pca with all components (6) -- ##
pca = PCA()

principal_components = pca.fit_transform(x_pca)
cumulative_sum_variance = np.cumsum(pca.explained_variance_ratio_)
cumulative_sum_variance = np.around(cumulative_sum_variance*100, 2)

st.write(f"The next chart shows the cumulative Variance explained by each of the 6 Principal Components")
st.write(f"""The 4 Principal Components alone explain {cumulative_sum_variance[4]}% 
of the Variance: it makes sense then, that the Clustering algorithms will be executed on the 4 principal components""")

fig = px.line(x=range(1, len(cumulative_sum_variance)+1), y=cumulative_sum_variance, 
                labels={"x":"Number of Components", "y":"Explained Variance (%)"}, text=cumulative_sum_variance,
                title="<b>Variance Accumulation", custom_data=[range(1,7), cumulative_sum_variance])
fig.update_layout(xaxis={"range":[0.5, len(cumulative_sum_variance)+0.5]},
                    yaxis={"range":[0,110]}, showlegend=False, hovermode="x")
fig.update_traces(textposition="top center", hovertemplate="Principal Components: %{customdata[0]}<br>Variance explained: %{customdata[1]}<extra></extra>")
fig.add_bar(x=list(range(1, len(cumulative_sum_variance)+1)), y=cumulative_sum_variance)
st.write(fig)

col1, col2 = st.columns(2)

with col1:
    ## -- 2 Pricipal Components -- ##
    st.subheader("2 Principal Components")

    pca_2 = PCA(n_components=2)
    pca_2.fit_transform(x_pca)
    x_pca_2 = pca_2.transform(x_pca)
    pca2_df = pd.DataFrame(x_pca_2, columns=["pc1", "pc2"])

    fig = px.scatter(x=x_pca_2[:,0], y=x_pca_2[:,1], labels={"x": "First Principal Component", "y":"Second Principal Component"},
                        title=f"<b>2D Scatterplot: {cumulative_sum_variance[1]}% of variance captured",
                        custom_data=custom_cols)
    fig.update_traces(hovertemplate=custom_template)
    st.write(fig)

with col2:
    ## -- 3 Pricipal Components -- ##
    st.subheader("3 Principal Components")

    pca_3 = PCA(n_components=3)
    pca_3.fit_transform(x_pca)
    x_pca_3 = pca_3.transform(x_pca)

    fig = px.scatter_3d(x=x_pca_3[:,0], y=x_pca_3[:,1], z=x_pca_3[:,2], 
                        labels={"x": "First Principal Component", "y":"Second Principal Component", "z":"Third Principal Component"},
                        title=f"<b>3D Scatterplot: {cumulative_sum_variance[2]}% of variance captured",
                        custom_data=custom_cols)
    fig.update_traces(marker_size = 3, hovertemplate=custom_template)
    st.write(fig)

## -- 4 Pricipal Components -- ##
# TODO: change n dinamically ?
st.subheader("Observation vs Transformed Data")

N = st.slider("Number of Principal Components:", 1, 6, 4)

pca_n = PCA(n_components=N)
principal_components = pca_n.fit_transform(x_pca)
pca4_df = pd.DataFrame(principal_components)

x_pca_n = pca_n.transform(x_pca) # variable for scatterplot visualization

fig = px.line(x_pca_n, labels={"index":"Observation", "value":"Transformed Data"},
                title=f"<b>Transformed Data by PCA: {cumulative_sum_variance[N-1]}% variance</b>")
# newnames = {'0':'PC1', '1': 'PC2', '2':'PC3', '3':'PC4'}
new_names = {}
for i in range(N):
    new_names[str(i)] = "PC"+str(i+1)

fig.for_each_trace(lambda t: t.update(name = new_names[t.name],
                                      legendgroup = new_names[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, new_names[t.name])
                                     )
                  )
fig.update_layout(legend_title="Principal Components")
st.write(fig)

st.write("---")

### --- Clustering Tendency --- ###

# TODO: VAT matrix needed: see if it can be executed in some more time


### --- K-Means --- ###
st.header("K-Means Clustering")

st.subheader("Choice of K")

## -- Elbow Method -- ##

K = range(1,50)
try:
    sum_squared_dist = pickle.load(open(pickle_dir+"sum_squared_dist", "rb"))
except (OSError, IOError) as e:
    sum_squared_dist = []
    for k in K:
        km = KMeans(n_clusters=k, random_state=0)
        km = km.fit(x_pca_n)
        sum_squared_dist.append(km.inertia_)
    pickle.dump(sum_squared_dist, open(pickle_dir+"sum_squared_dist", "wb"))
kn = KneeLocator(K, sum_squared_dist, curve='convex', direction='decreasing')

col1, col2 = st.columns(2)

with col1:
    fig = px.line(x=K, y=sum_squared_dist, labels={"x":"number of clusters - k", "y":"Sum of squared distances"},
                title="<b>Clustering Quality - Elbow Point")
    fig.add_vline(x=kn.knee, line_width=1.5, line_dash="dash", line_color="green", annotation_text=f"Elbow point: {kn.knee}")
    fig.update_layout(xaxis={"range":[min(K)-1,max(K)+1]})
    st.write(fig)

## -- Silhouette Score -- ##
K = range(2,15)
try:
    silhouette_list = pickle.load(open(pickle_dir+"silhouette_list", "rb"))
except (OSError, IOError) as e:
    silhouette_list = []
    for k in K:
        km = KMeans(n_clusters=k, random_state=0)
        km_labels = km.fit_predict(x_pca_n)
        silhouette_avg = silhouette_score(x_pca_n, km_labels)
        silhouette_list.append(silhouette_avg)
    pickle.dump(silhouette_list, open(pickle_dir+"silhouette_list", "wb"))

with col2:
    fig = px.line(x=K, y=silhouette_list, labels={"x":"Number of clusters - k", "y":"Silhouette Score"},
                    title="<b>Clustering Quality - Silhouette Score", markers=True)
    fig.update_layout(xaxis={"dtick":2, "range":[min(K)-1,max(K)+1]})
    st.write(fig)

# ## -- K Evaluation -- ##
# K = [2, 3, 4, 5]

# try:
#     kms = pickle.load(open("kms", "rb"))
# except (OSError, IOError) as e:
#     kms = []
#     for i in K:
#         # Create KMeans instance for different number of clusters
#         kms.append(KMeans(n_clusters=i, random_state=0))
#     pickle.dump(kms, open(pickle_dir+"kms", "wb"))

# fig, ax = plt.subplots(2, 2, figsize=big_size)
# k = 2
# for i in K:
#     km = kms[k-2]
#     q, mod = divmod(k, 2)

#     # Create SilhouetteVisualizer instance with KMeans instance and Fit the visualizer
#     visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
#     visualizer.fit(x_pca_4)

#     ax[q-1][mod].set_title(f"Silhouette Score for k={i}")
#     ax[q-1][mod].set_xlabel("Silhouette Score")
#     ax[q-1][mod].set_ylabel("Istances")

#     k+=1
# plt.savefig(buffer, format="png")
# st.image(buffer)

## -- K-Means with K=3 -- ##
st.subheader("K-Means with K=3")

try:
    km3 = pickle.load(open(pickle_dir+"km3", "rb"))
except (OSError, IOError) as e:
    km3 = KMeans(n_clusters=3, random_state=0).fit(x_pca_n)
    pickle.dump(km3, open(pickle_dir+"km3", "wb"))
colors = color_palette_mapping(km3.labels_)

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(x=x_pca_2[:,0], y=x_pca_2[:,1], labels={"x": "First Principal Component", "y":"Second Principal Component"},
                        title=f"<b>K-Means Clustering plotted on 2 Principal Components (K=3)",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)

with col2:
    fig = px.scatter_3d(x=x_pca_3[:,0], y=x_pca_3[:,1], z=x_pca_3[:,2], 
                        labels={"x": "First Principal Component", "y":"Second Principal Component", "z":"Third Principal Component"},
                        title=f"<b>K-Means Clustering plotted on 3 Principal Components (K=3)",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(marker_size = 3, hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)

st.write("---")

### --- DBSCAN Clustering --- ###
st.header("DBSCAN Clustering")

try:
    dbscan = pickle.load(open(pickle_dir+"dbscan", "rb"))
except (OSError, IOError) as e:
    dbscan = DBSCAN().fit(x_pca_n)
    pickle.dump(dbscan, open(pickle_dir+"dbscan", "wb"))
colors = color_palette_mapping(dbscan.labels_)

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(x=x_pca_2[:,0], y=x_pca_2[:,1], labels={"x": "First Principal Component", "y":"Second Principal Component"},
                        title=f"<b>DBSCAN Clustering plotted on 2 Principal Components",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)

with col2:
    fig = px.scatter_3d(x=x_pca_3[:,0], y=x_pca_3[:,1], z=x_pca_3[:,2], 
                        labels={"x": "First Principal Component", "y":"Second Principal Component", "z":"Third Principal Component"},
                        title=f"<b>DBSCAN Clustering plotted on 3 Principal Components",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(marker_size = 3, hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)

st.write("---")

### --- OPTICS Clustering --- ###
st.header("OPTICS Clustering")
try:
    optics = pickle.load(open(pickle_dir+"optics", "rb"))
except (OSError, IOError) as e:
    optics = OPTICS().fit(x_pca_n)
    pickle.dump(optics, open(pickle_dir+"optics", "wb"))
colors = color_palette_mapping(optics.labels_)

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(x=x_pca_2[:,0], y=x_pca_2[:,1], labels={"x": "First Principal Component", "y":"Second Principal Component"},
                        title=f"<b>OPTICS Clustering plotted on 2 Principal Components",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)
with col2:
    fig = px.scatter_3d(x=x_pca_3[:,0], y=x_pca_3[:,1], z=x_pca_3[:,2], 
                        labels={"x": "First Principal Component", "y":"Second Principal Component", "z":"Third Principal Component"},
                        title=f"<b>OPTICS Clustering plotted on 3 Principal Components",
                        custom_data=custom_cols, color=colors)
    fig.update_traces(marker_size = 3, hovertemplate=custom_template)
    fig.update_layout(showlegend=False)
    st.write(fig)
