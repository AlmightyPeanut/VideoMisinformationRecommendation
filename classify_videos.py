import os.path
from os import listdir
import json

import numpy as np
import pandas as pd
import tqdm
from keras.utils import to_categorical

from classifier.config.ClassifierConfig import Config
from pseudoscientificvideosdetection.PseudoscienceClassifier import PseudoscienceClassifier

VIDEO_DIR = 'data/video_data'

if __name__ == '__main__':
    with open('data/youtube_ids.json', 'r') as f:
        youtube_ids = json.load(f)['all']

    pseudoscienceClassifier = PseudoscienceClassifier()

    predictions = []
    for video_id in tqdm.tqdm(youtube_ids):
        path = f'{VIDEO_DIR}/{video_id}/{video_id}.json'
        if not os.path.exists(path):
            continue

        with open(os.path.join(path), 'r') as f:
            video = json.load(f)
        if not video:
            continue

        prediction = pseudoscienceClassifier.classify(video_details=video)

        predictions.append({
            "video_id": video_id,
            "label": prediction[0],
            "confidence": prediction[1]
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("data/predictions.csv", index=False)
