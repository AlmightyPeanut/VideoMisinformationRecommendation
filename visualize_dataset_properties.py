import os
import pandas as pd

if __name__ == '__main__':
    all_used_videos_df = pd.read_csv('data/predictions.csv', index_col=0)
    print("#Collected videos: ", all_used_videos_df.shape[0])
    print("#Neutral videos: ", all_used_videos_df[all_used_videos_df['label'] == 'neutral'].shape[0])
    print("#Promoting videos: ", all_used_videos_df[all_used_videos_df['label'] == 'promoting'].shape[0])
    print("#Debunking videos: ", all_used_videos_df[all_used_videos_df['label'] == 'debunking'].shape[0])

    seed_data = pd.read_csv('data/yaudit-data/train.csv', index_col=0)
    amount_of_seed_videos = seed_data.index.isin(all_used_videos_df.index).shape[0]
    print("#Seed videos: ", amount_of_seed_videos)
    print("#Automatically annotated videos: ", all_used_videos_df.shape[0] - amount_of_seed_videos)

    empty_comments_files = 0
    for video_id in os.listdir('data/video_data'):
        comments_file_path = os.path.join('data/video_data', video_id, video_id + '_comments.json')
        if not os.path.exists(comments_file_path):
            empty_comments_files += 1
            continue
        with open(comments_file_path, 'r') as f:
            if len(f.read()) == 0:
                empty_comments_files += 1
    print("#Disabled comments: ", empty_comments_files)
