import json
import multiprocessing
import os
import pickle
from functools import partial

import fasttext
import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from dataset.DatasetUtils import DatasetUtils

SOURCE_DIR = '../AuditRepo.nosync/yaudit-papadamou-model/yaudit-data/comments'
VIDEO_BASE_DIR = 'data/video_data'


def feature_index_mapping(data: pd.DataFrame):
    index_mapping = {}
    feature_mapping = pd.Index(data.values.flatten()).unique()

    for item_index, features in tqdm.tqdm(data.iterrows()):
        index_mapping[item_index] = features.map(feature_mapping.get_loc).values

    return index_mapping, feature_mapping


if __name__ == '__main__':
    with open('data/rec_sys_dataset.pkl', 'rb') as f:
        df = pickle.load(f).T

    # users
    home_page = pd.read_csv('data/yaudit-data/home_page_results.csv')
    recommendations = pd.read_csv('data/yaudit-data/recommendations.csv')
    search_results = pd.read_csv('data/yaudit-data/search_results.csv')

    youtube_ids = pd.concat([home_page.youtube_id, recommendations.youtube_id,
                             recommendations.watched_youtube_id, search_results.youtube_id])
    user_ids = pd.concat([home_page.bot_id, recommendations.bot_id,
                          recommendations.bot_id, search_results.bot_id])
    timestamps = pd.concat([home_page.started_at, recommendations.started_at,
                            recommendations.started_at, search_results.started_at])
    youtube_id_user_mapping = pd.concat([youtube_ids, user_ids, timestamps], axis=1)
    youtube_id_user_mapping.columns = ['youtube_id', 'user_id', 'timestamp']

    # items
    youtube_id_mapping = pd.DataFrame(df.index.unique()).reset_index().set_index(0)['index']

    # watched youtube_ids
    watched_youtube_id_user_mapping = pd.concat([recommendations.watched_youtube_id,
                                                 recommendations.bot_id,
                                                 recommendations.started_at], axis=1).drop_duplicates(ignore_index=True)
    watched_youtube_id_user_mapping.columns = ['youtube_id', 'user_id', 'timestamp']

    # recommended to the user, but no watched
    recommended_youtube_id_user_mapping = pd.concat([recommendations.youtube_id,
                                                     recommendations.bot_id,
                                                     recommendations.started_at], axis=1).drop_duplicates(ignore_index=True)
    recommended_youtube_id_user_mapping.columns = ['youtube_id', 'user_id', 'timestamp']
    # remove duplicate entries with the watched mapping
    recommended_youtube_id_user_mapping = recommended_youtube_id_user_mapping.merge(
        watched_youtube_id_user_mapping[['youtube_id', 'user_id']].assign(vec=True),
        how='left', on=['youtube_id', 'user_id']
    ).fillna(False)
    recommended_youtube_id_user_mapping = recommended_youtube_id_user_mapping[
        recommended_youtube_id_user_mapping['vec'] == False].drop('vec', axis=1).reset_index(drop=True)

    # exporting base data
    export_df = []
    for youtube_id in tqdm.tqdm(df.index[:]):
        item_id = youtube_id_mapping.loc[youtube_id]
        filtered_mapping = youtube_id_user_mapping[youtube_id_user_mapping.youtube_id == youtube_id]
        watched_index = watched_youtube_id_user_mapping[
            ((watched_youtube_id_user_mapping.youtube_id == youtube_id)
             & (watched_youtube_id_user_mapping.user_id.isin(filtered_mapping.user_id))
             & (watched_youtube_id_user_mapping.timestamp.isin(filtered_mapping.timestamp)))
        ]

        recommended_not_watched_index = recommended_youtube_id_user_mapping[
            ((recommended_youtube_id_user_mapping.youtube_id == youtube_id)
             & (recommended_youtube_id_user_mapping.user_id.isin(filtered_mapping.user_id))
             & (recommended_youtube_id_user_mapping.timestamp.isin(filtered_mapping.timestamp)))
        ]

        def assign_new_rating_for_index(original, index, rating):
            index = index.assign(new_rating=rating)
            original = pd.merge(original, index, on=['youtube_id', 'user_id', 'timestamp'], how='left')
            original['rating'] = original['new_rating'].fillna(original['rating'])
            original = original.drop('new_rating', axis=1)
            return original

        filtered_mapping = filtered_mapping.assign(rating=0)
        if not watched_index.empty:
            filtered_mapping = assign_new_rating_for_index(filtered_mapping, watched_index, 1)
        if not recommended_not_watched_index.empty:
            filtered_mapping = assign_new_rating_for_index(filtered_mapping, recommended_not_watched_index, 0)

        filtered_mapping.loc[:, 'youtube_id'] = item_id
        export_df.append(filtered_mapping)

    export_df = pd.concat(export_df).drop_duplicates()

    # elliot doesn't handle equal entries with different timestamps,
    # so we sort this out ourselves by only leaving the entries with the highest values
    # TODO: or average them or sth else?
    export_df = export_df.sort_values(by=['youtube_id', 'user_id', 'rating'], ascending=[True, True, False])
    export_df = export_df.drop_duplicates(subset=['youtube_id', 'user_id'], keep='first')
    export_df.reset_index(drop=True, inplace=True)

    export_df.to_csv('data/base_data.tsv', sep='\t', index=False, header=False,
                     columns=["user_id", "youtube_id", "rating", "timestamp"])

    df.index = df.index.map(lambda yt_id: youtube_id_mapping[yt_id])

    # apply mapping to the predictions
    prediction_df = pd.read_csv('data/predictions.csv')
    prediction_df['video_id'] = prediction_df['video_id'].map(lambda yt_id: youtube_id_mapping[yt_id])
    prediction_df.to_csv('data/predictions_with_item_id_mapping.tsv', sep='\t', index=False, header=True)

    # export the word embeddings
    index_mapping, feature_mapping = feature_index_mapping(df.iloc[:])
    pd.DataFrame(index_mapping).T.to_csv('data/rec_sys_feature_map.tsv', sep='\t', index=True, header=False)
    feature_mapping.to_series(list(range(feature_mapping.size))).to_csv('data/rec_sys_features.tsv', sep='\t',
                                                                        index=True, header=False)
    df.iloc[:].to_csv('data/rec_sys_dataset.tsv', sep='\t', index=True, header=False)

