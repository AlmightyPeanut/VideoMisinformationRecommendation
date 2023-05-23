import json
import multiprocessing
import os
import pickle

import fasttext
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from dataset.DatasetUtils import DatasetUtils

SOURCE_DIR = '../AuditRepo.nosync/yaudit-papadamou-model/yaudit-data/comments'
VIDEO_BASE_DIR = 'data/video_data'

if __name__ == '__main__':
    with open('data/rec_sys_dataset.pkl', 'rb') as f:
        df = pickle.load(f).T

    classes_df = pd.read_csv('data/predictions.csv', index_col='video_id')

    dimensions = 2
    pca = PCA(n_components=dimensions)
    components = pca.fit_transform(df)
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    columns = list(labels.values())
    data = pd.DataFrame(components, columns=columns, index=df.index)

    print(labels.values())

    fig = px.scatter(
        data,
        x=columns[0],
        y=columns[1],
        labels=labels,
        color=classes_df['label']
    )
    # fig.update_traces(diagonal_visible=False)
    fig.show()
