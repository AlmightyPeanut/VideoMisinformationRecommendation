import json
import os

import fasttext
import numpy as np
import pandas as pd
import tqdm

from dataset.DatasetUtils import DatasetUtils

VIDEO_DIR = 'data/video_data'
FASTTEXT_MODELS_DIR = 'pseudoscientificvideosdetection/models/feature_extraction'


def main():
    DATASET = DatasetUtils()
    # Load FastText Classifiers
    if not os.path.isfile(f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_snippet.bin'):
        exit('Cannot find fasttext feature extractor for VIDEO SNIPPET')
    FASTTEXT_VIDEO_SNIPPET = fasttext.load_model(path=f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_snippet.bin')
    if not os.path.isfile(f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_transcript.bin'):
        exit('Cannot find fasttext feature extractor for VIDEO TRANSCRIPT')
    FASTTEXT_VIDEO_TRANSCRIPT = fasttext.load_model(path=f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_transcript.bin')
    if not os.path.isfile(f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_comments.bin'):
        exit('Cannot find fasttext feature extractor for VIDEO COMMENTS')
    FASTTEXT_VIDEO_COMMENTS = fasttext.load_model(path=f'{FASTTEXT_MODELS_DIR}/fasttext_model_video_comments.bin')

    with open('data/youtube_ids.json', 'r') as f:
        youtube_ids = json.load(f)['all']

    rec_sys_data = {}
    for video_id in tqdm.tqdm(youtube_ids):
        path = f'{VIDEO_DIR}/{video_id}/{video_id}.json'
        if not os.path.exists(path):
            continue

        with open(os.path.join(path), 'r') as f:
            video_details = json.load(f)
        if not video_details:
            continue

        """ Prepare Classifier Input """
        # --- VIDEO SNIPPET
        video_snippet = '{} {}'.format(video_details['snippet']['title'], video_details['snippet']['description'])
        video_snippet = DATASET.preprocess_text(text=video_snippet)
        X_video_snippet = FASTTEXT_VIDEO_SNIPPET.get_sentence_vector(text=video_snippet)

        # --- VIDEO TRANSCRIPT
        video_transcript = DATASET.read_video_transcript(video_id=video_details['id'])
        video_transcript_processed = DATASET.preprocess_video_transcript(video_captions=video_transcript)
        X_video_transcript = FASTTEXT_VIDEO_TRANSCRIPT.get_sentence_vector(text=video_transcript_processed)

        # --- VIDEO COMMENTS
        video_comments = DATASET.read_video_comments(video_id=video_details['id'])
        video_comments_preprocessed = DATASET.preprocess_video_comments(video_comments=video_comments)
        X_video_comments = FASTTEXT_VIDEO_COMMENTS.get_sentence_vector(text=' '.join(video_comments_preprocessed))

        data = np.concatenate(([X_video_snippet.reshape(1, -1)],
                               [X_video_transcript.reshape(1, -1)],
                               [X_video_comments.reshape(1, -1)]), axis=None)
        rec_sys_data[video_id] = {f'video_word_vector_feature_{i}': value for i, value in enumerate(data)}
    pd.DataFrame.from_dict(rec_sys_data).T.to_pickle('data/rec_sys_dataset.pkl')


if __name__ == '__main__':
    main()
