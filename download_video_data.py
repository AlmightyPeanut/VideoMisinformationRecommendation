import sys

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from socket import error as SocketError
from tqdm import tqdm
import time
import glob
import subprocess
import pandas as pd
import os
import json

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_API_KEY = 'AIzaSyATXGPGvr9kcl3Lr7KtiGN5i1D-1a0jsfM'
YOUTUBE_API = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

LIMIT_PAGES_COMMENTS = 1

VIDEO_BASE_DIR = 'data/video_data'


def download_video_metadata(video_id):
    while True:
        try:
            # Send request to get video's information
            response = YOUTUBE_API.videos().list(
                part='id,snippet,contentDetails,statistics,status',
                id=video_id
            ).execute()

            # Get Video Details
            try:
                return response['items'][0]
            except IndexError:
                if "status" not in response:
                    return None
            except Exception as e:
                print(e)
                sys.exit(0)

        except (HttpError, SocketError) as error:
            print(f'--- HTTP Error occurred while retrieving information for VideoID: {video_id}. [ERROR]: {error}')
            time.sleep(30)


def is_video_transcript_downloaded(video_id):
    video_transcript = glob.glob(f'{VIDEO_BASE_DIR}/{video_id}/{video_id}_transcript.*')
    if len(video_transcript) > 0:
        return True
    return False


def download_video_transcript(video_id):
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}_transcript'

    try:
        # download_video_transcript.sh needs to have execute permissions
        output = subprocess.check_output(
            f'bash yaudit_papadamou/youtubescripts/download_video_transcript.sh {video_url} {path}', shell=True)
        if "HTTP_ERROR" in str(output) or "There's no subtitles for the requested languages" in str(output):
            open(path + '.en.vtt', 'a').close()
            return
    except subprocess.CalledProcessError as e:
        pass
    return


def is_blocked_in_region(video_details):
    if 'regionRestriction' in video_details['contentDetails']:
        if (('blocked' in video_details['contentDetails']['regionRestriction']
             and 'NL' in video_details['contentDetails']['regionRestriction']['blocked'])
                or ('allowed' in video_details['contentDetails']['regionRestriction']
                    and 'NL' not in video_details['contentDetails']['regionRestriction']['allowed'])):
            # print("Video unavailable")
            return True
    return False


def main():
    with open("data/youtube_ids.json", 'r') as f:
        videos = json.load(f)
    youtube_ids = videos['all']

    if not os.path.exists(VIDEO_BASE_DIR):
        os.mkdir(VIDEO_BASE_DIR)

    unavailable_videos_path = f'data/unavailable_videos.json'
    if not os.path.exists(unavailable_videos_path):
        PRIVATE_VIDEOS = []
        NOT_AVAILABLE_IN_COUNTRY = []
    else:
        with open(unavailable_videos_path, 'r') as f:
            unavailable_videos_dict = json.load(f)
        PRIVATE_VIDEOS = unavailable_videos_dict['private_videos']
        NOT_AVAILABLE_IN_COUNTRY = unavailable_videos_dict['country_restriction']

    for video_id in tqdm(youtube_ids):
        if video_id in PRIVATE_VIDEOS or video_id in NOT_AVAILABLE_IN_COUNTRY:
            continue

        os.makedirs(f'{VIDEO_BASE_DIR}/{video_id}', exist_ok=True)
        video_details_path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}.json'

        # Download video details
        if not os.path.exists(video_details_path):
            video_details = download_video_metadata(video_id)
            if video_details is None:
                PRIVATE_VIDEOS.append(video_id)
                video_details = ""

            if video_details != "" and is_blocked_in_region(video_details):
                NOT_AVAILABLE_IN_COUNTRY.append(video_id)
                video_details = ""

            with open(video_details_path, 'w') as f:
                json.dump(video_details, f)

        # download video transcript
        if not is_video_transcript_downloaded(video_id):
            download_video_transcript(video_id)

        # download video comments
        comment_path = f'{VIDEO_BASE_DIR}/{video_id}/{video_id}_comments.json'
        if not os.path.exists(comment_path):
            os.system(
                "python3 yaudit_papadamou/youtubescripts/download_youtube_video_comments.py {0} {1} {2} {3}".format(
                    video_id, VIDEO_BASE_DIR, LIMIT_PAGES_COMMENTS, YOUTUBE_API_KEY))

        time.sleep(1)
    print("Finished downloading.")

    with open(unavailable_videos_path, 'w') as f:
        json.dump({
            "private_videos": PRIVATE_VIDEOS,
            "country_restriction": NOT_AVAILABLE_IN_COUNTRY
        }, f)
    print("#private videos: ", len(set(PRIVATE_VIDEOS)))
    print("#geo-restricted videos: ", len(set(NOT_AVAILABLE_IN_COUNTRY)))


if __name__ == '__main__':
    main()
