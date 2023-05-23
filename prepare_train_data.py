import glob
import json
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
import tqdm
from nltk import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords

VIDEO_BASE_DIR = 'data/video_data'


def get_video_transcript(path):
    transcript_path = glob.glob(path)
    if len(transcript_path) == 0:
        return ''

    transcript_path = transcript_path[0]
    with open(transcript_path, 'r') as f:
        return f.read()


def process_train_video_metadata(path):
    with open(path, 'r') as f:
        video = json.load(f)
        if video == '':
            return {}

        return {
            'uuid': video['id'],
            'channel_id': video['snippet']['channelId'],
            'category_id': video['snippet']['categoryId'],
            'default_audio_language': video['snippet']['defaultAudioLanguage'] if 'defaultAudioLanguage' in video[
                'snippet'] else np.nan,
            'description': video['snippet']['description'],
            'title': video['snippet']['title'],
            'published_at': pd.Timestamp(video['snippet']['publishedAt']),
            'dislike_count': video['statistics']['dislikeCount'] if 'dislikeCount' in video['statistics'] else np.nan,
            'favourite_count': video['statistics']['favoriteCount'] if 'favoriteCount' in video[
                'statistics'] else np.nan,
            'comment_count': video['statistics']['commentCount'] if 'commentCount' in video['statistics'] else np.nan,
            'view_count': video['statistics']['viewCount'] if 'viewCount' in video['statistics'] else np.nan,
            'like_count': video['statistics']['likeCount'] if 'likeCount' in video['statistics'] else np.nan,
        }


def remove_tags(text):
    """
    Remove vtt markup tags
    """
    tags = [
        r'</c>',
        r'<c(\.color\w+)?>',
        r'<\d{2}:\d{2}:\d{2}\.\d{3}>',

    ]

    for pat in tags:
        text = re.sub(pat, '', text)

    # extract timestamp, only kep HH:MM
    text = re.sub(
        r'(\d{2}:\d{2}):\d{2}\.\d{3} --> .* align:start position:0%',
        r'\g<1>',
        text
    )

    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    return text


def remove_header(lines):
    """
    Remove vtt file header
    """
    pos = -1
    for mark in ('##', 'Language: en',):
        if mark in lines:
            pos = lines.index(mark)
    lines = lines[pos + 1:]
    return lines


def merge_duplicates(lines):
    """
    Remove duplicated subtitles. Duplacates are always adjacent.
    """
    last_timestamp = ''
    last_cap = ''
    for line in lines:
        if line == "":
            continue
        if re.match('^\d{2}:\d{2}$', line):
            if line != last_timestamp:
                last_timestamp = line
        else:
            if line != last_cap:
                yield line
                last_cap = line


def merge_short_lines(lines):
    buffer = ''
    for line in lines:
        if line == "" or re.match('^\d{2}:\d{2}$', line):
            yield '\n' + line
            continue

        if len(line + buffer) < 80:
            buffer += ' ' + line
        else:
            yield buffer.strip()
            buffer = line
    yield buffer


def parse_transcript(text):
    text = remove_tags(text)
    lines = text.splitlines()
    lines = remove_header(lines)
    lines = merge_duplicates(lines)
    lines = list(lines)
    lines = merge_short_lines(lines)
    lines = list(lines)
    result = ' '.join(lines)
    return re.sub('\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3} ', '', result)


def transform_video(video):
    return {
        "annotation": {
            "annotations": [
            ],
            "label": video['annotation'],
            "manual_review_label": video['annotation']
        },
        "etag": "",
        "id": video['uuid'],
        "isSeed": True,
        "kind": "youtube#video",
        "relatedVideos": [
        ],
        "search_term": "",
        "snippet": {
            "categoryId": video['category_id'] if video['category_id'] else "",
            "channelId": video['channel_id'] if video['channel_id'] else "",
            "channelTitle": "",
            "defaultAudioLanguage": video['default_audio_language'] if video['default_audio_language'] else "",
            "description": video['description'],
            "liveBroadcastContent": "none",
            "localized": {
                "description": video['description'],
                "title": video['title']
            },
            "publishedAt": video['published_at'].isoformat(),
            "tags": [
            ],
            "thumbnails": {
            },
            "title": video['title']
        },
        "statistics": {
            "commentCount": video['comment_count'] if video['comment_count'] else 0,
            "dislikeCount": video['dislike_count'] if video['dislike_count'] else 0,
            "favoriteCount": video['favourite_count'] if video['favourite_count'] else 0,
            "likeCount": video['like_count'] if video['like_count'] else 0,
            "viewCount": video['view_count'] if video['view_count'] else 0
        }
    }


def transform_video_to_transcript(video):
    return {
        "captions": [
            sent.lower()
            for sent in sent_tokenize(video['clean_transcript'])
        ],
        "id": video['uuid']
    }


def process_comment(comment, video_id, can_reply, is_public, reply_count) -> dict:
    return {
        'id': comment['id'],
        'uuid': comment['id'],
        'video_uuid': video_id,
        'author_display_name': comment['snippet']['authorDisplayName'],
        'author_profile_image_url': comment['snippet']['authorProfileImageUrl'],
        'author_channel_url': comment['snippet']['authorChannelUrl'],
        'text_display': comment['snippet']['textDisplay'],
        'text_original': comment['snippet']['textOriginal'],
        'parent_uuid': comment['snippet']['parentId'] if 'parentId' in comment['snippet'] else '',
        'can_rate': comment['snippet']['canRate'],
        'like_count': comment['snippet']['likeCount'],
        'published_at': pd.to_datetime(comment['snippet']['publishedAt']),
        'youtube_update_timestamp': pd.to_datetime(comment['snippet']['updatedAt']),
        'can_reply': can_reply,
        'total_reply_count': reply_count,
        'is_public': is_public,
    }


def process_train_video_comments(path):
    with open(path, 'r') as f:
        data = f.read().split('\n')
        data = '[' + ','.join(data[:-1]) + ']'
        comment_threads = json.loads(data)

        processed_comments = []
        for comment_thread in comment_threads:
            video_id = comment_thread['snippet']['videoId']
            can_reply = comment_thread['snippet']['canReply']
            is_public = comment_thread['snippet']['isPublic']
            reply_count = comment_thread['snippet']['totalReplyCount']
            processed_comments.append(process_comment(comment_thread['snippet']['topLevelComment'],
                                                      video_id, can_reply, is_public, reply_count))

        return processed_comments


def transform_comment(video_uuid, comment_ids):
    return {
        "comments": comment_ids,
        "id": video_uuid
    }


def comment_to_youtube_response(comment):
    return {
        'snippet': {
            'topLevelComment': {
                'snippet': {
                    'authorDisplayName': comment['author_display_name'],
                    'authorProfileImageUrl': comment['author_profile_image_url'],
                    'authorChannelUrl': comment['author_channel_url'],
                    'textDisplay': comment['text_display'],
                    'textOriginal': comment['text_original'],
                    'parentId': comment['parent_uuid'],
                    'canRate': comment['can_rate'],
                    'likeCount': comment['like_count'],
                    'publishedAt': comment['published_at'].isoformat(),
                    'updatedAt': comment['youtube_update_timestamp'].isoformat(),
                    # 'authorChannelId': {
                    #     'value': comment['author_channel_uuid']
                    # }
                }
            },
            'canReply': comment['can_reply'],
            'totalReplyCount': comment['total_reply_count'],
            'isPublic': comment['is_public']
        }
    }


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    return " ".join(filtered_words)


def main():
    videos = pd.read_csv('data/train.csv')
    youtube_ids = list(videos['youtube_id'])

    # Read video data
    print("Processing video data and transcripts...")
    processed_train_videos = []
    for video_id in youtube_ids:
        path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}.json'
        transcript_path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}_transcript.*'
        if os.path.exists(path):
            processed_video = process_train_video_metadata(path)
            if len(processed_video) == 0:
                continue
            processed_video['transcript'] = get_video_transcript(transcript_path)
            processed_video['annotation'] = videos[videos['youtube_id'] == video_id]['annotation'].values[0]

            processed_train_videos.append(processed_video)
    videos = pd.DataFrame(processed_train_videos)

    # Process transcripts
    videos['transcript'] = videos['transcript'].fillna('')
    videos['clean_transcript'] = videos['transcript'].apply(parse_transcript)

    # Remove unknown annotations and duplicates
    videos = videos[(videos['annotation'] != 'unknown') & ~videos['annotation'].isna()]
    videos.drop_duplicates(subset='uuid', keep='first', inplace=True)

    videos['comment_count'] = videos['comment_count'].fillna(0)
    videos['dislike_count'] = videos['dislike_count'].fillna(0)
    videos['like_count'] = videos['like_count'].fillna(0)
    videos['favourite_count'] = videos['favourite_count'].fillna(0)
    videos['view_count'] = videos['view_count'].fillna(0)
    videos['default_audio_language'] = videos['default_audio_language'].fillna('')

    print("Exporting groundtruth dataset...")
    with open('data/groundtruth_dataset.json', 'w') as f:
        f.write("[")
        for i, video in videos.iterrows():
            data = json.dumps(transform_video(video))
            f.write(f'{data},\n')
        f.seek(f.tell() - 2, os.SEEK_SET)
        f.truncate()
        f.write("]")

    with open('data/groundtruth_videos_transcripts.json', 'w') as f:
        f.write("[")
        for _, video in videos.iterrows():
            data = json.dumps(transform_video_to_transcript(video))
            f.write(data + ',\n')
        f.seek(f.tell() - 2, os.SEEK_SET)
        f.truncate()
        f.write("]")

    # Process comments
    print("Processing comments...")
    processed_train_comments = []
    for video_id in tqdm.tqdm(youtube_ids):
        comment_path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}_comments.json'
        if os.path.exists(comment_path):
            processed_train_comments += process_train_video_comments(comment_path)
    comments = pd.DataFrame(processed_train_comments)
    comments.to_pickle('data/comments.p')

    # Create FastText train data
    nltk.download('stopwords')

    print('Exporting fasttext training data...')
    FASTTEXT_TRAIN_DIR = 'data/fasttext_data'
    os.makedirs(FASTTEXT_TRAIN_DIR, exist_ok=True)

    with open('data/groundtruth_dataset.json', 'r') as f:
        with open(f'{FASTTEXT_TRAIN_DIR}/video_snippet_train_data.txt', 'w') as output:
            dataset = json.load(f)
            for data in dataset:
                sentence = preprocess(data['snippet']['description'])
                output.write(
                    sentence + '\n'
                )

    comments = pd.read_pickle('data/comments.p')
    with open(f'{FASTTEXT_TRAIN_DIR}/video_comments_train_data.txt', 'w') as output:
        for comment in comments['text_original'].sample(100000):
            sentence = preprocess(comment)
            output.write(sentence + '\n')

    with open('data/groundtruth_videos_transcripts.json', 'r') as f:
        with open(f'{FASTTEXT_TRAIN_DIR}/video_transcript_train_data.txt', 'w') as output:
            dataset = json.load(f)
            for data in dataset:
                for sentence in data['captions']:
                    sentence = preprocess(sentence)
                    output.write(sentence + '\n')

    with open('data/groundtruth_dataset.json', 'r') as f:
        with open(f'{FASTTEXT_TRAIN_DIR}/video_tags_train_data.txt', 'w') as output:
            dataset = json.load(f)
            for data in dataset:
                tags = data['snippet']['tags']
                if len(tags) > 0:
                    output.write(
                        ' '.join(tags) + '\n'
                    )

    print('Done!')


if __name__ == '__main__':
    main()
