import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from visualize_experiment_results import RESULTS_INDEX_MAPPING, BASELINE_ALGORITHMS, NEAREST_NEIGHBOUR_ALGORITHMS, \
    COLABORATIVE_FILTERING_ALGORITHMS, FACTORISATION_MACHINES_ALGORITHMS, NEURAL_ALGORITHMS

RESULTS_FOLDERS = [
    'results/yaudit_non_hybrid/recs',
    # 'results/yaudit_svd/recs',
]

TEST_FILE = 'data/splitting/non_hybrid/0/test.tsv'

FEATURES_TO_VISUALISE_NAMES = [
    '#Views',
    '#Likes',
    '#Comments',
    'Duration in Minutes',
]

FEATURES_TO_VISUALISE = [
    'view_count',
    'like_count',
    'comment_count',
    'duration_minutes',
]

if __name__ == '__main__':
    # 1. Get the relevant items from the test set for each user
    test_data = pd.read_csv(TEST_FILE, sep='\t', header=None)

    # 2. Get the recommendation of each algorithm
    result_files_dict = {}
    for results_folder in RESULTS_FOLDERS:
        for file_name in os.listdir(results_folder):
            if not file_name.endswith('.tsv'):
                continue
            algorithm_name = file_name.split('.')[0].split('_')[0]
            if algorithm_name not in result_files_dict:
                result_files_dict[algorithm_name] = []
            result_files_dict[algorithm_name].append(os.path.join(results_folder, file_name))


    def filter_for_most_recent_file(files: list[str]):
        most_recent_time = os.path.getmtime(files[0])
        file = files[0]
        for f in files[1:]:
            file_time = os.path.getmtime(f)
            if most_recent_time < file_time:
                most_recent_time = file_time
                file = f
        return file


    result_files_dict = dict(zip(result_files_dict.keys(),
                                 list(map(filter_for_most_recent_file, result_files_dict.values()))))

    algorithm_recommendations_dict = {}
    for algorithm_name, file_name in result_files_dict.items():
        algorithm_recommendations = pd.read_csv(file_name, sep='\t', header=None).drop(2, axis=1)
        algorithm_recommendations.columns = ['User', 'Item']

        unique_user_amount = len(algorithm_recommendations['User'].unique())
        algorithm_recommendations['Position'] = list(range(0, algorithm_recommendations.shape[0]
                                                           // unique_user_amount)) * unique_user_amount
        algorithm_recommendations_dict[algorithm_name] = algorithm_recommendations

    all_algorithm_recommendations = pd.concat([value for key, value in algorithm_recommendations_dict.items()],
                                              keys=algorithm_recommendations_dict.keys())
    all_algorithm_recommendations = all_algorithm_recommendations.reset_index(level=0, names='Algorithm')

    # 3. Visualise recommendation features
    item_features = pd.read_csv('data/yaudit-data/videos_metadata_processed.csv')
    youtube_id_mapping = pd.read_csv('data/base_data/youtube_id_mapping.csv')
    youtube_id_mapping.columns = ['youtube_id', 'item_id']
    index_list = ['item_id']
    index_list.extend(FEATURES_TO_VISUALISE)
    item_features = pd.merge(item_features, youtube_id_mapping, how='left', on='youtube_id')[index_list]
    item_features = item_features[~item_features['item_id'].isna()].fillna(0)
    new_columns = ['Item']
    new_columns.extend(FEATURES_TO_VISUALISE_NAMES)
    item_features.columns = new_columns
    all_algorithm_recommendations_features = pd.merge(all_algorithm_recommendations, item_features,
                                                      how='left', on='Item')
    all_algorithm_recommendations_features['Algorithm'] = all_algorithm_recommendations_features['Algorithm'].map(
        RESULTS_INDEX_MAPPING.get)

    algorithm_sorting = ['SVD++', 'U-k NN', 'I-k NN', 'DMF', 'NNMF', 'BPRMF', 'MostPop', 'PMF',
                         'GMF', 'FunkSVD', 'FFM', 'LogMF', 'CML', 'MF', 'NeuMF', 'Random']
    algorithm_sorting = dict(zip(algorithm_sorting, list(range(0, len(algorithm_sorting)))))
    all_algorithm_recommendations_features.sort_values(by='Algorithm', key=lambda x: x.map(algorithm_sorting),
                                                       inplace=True)

    # color palette per type of algorithm
    algorithm_names = all_algorithm_recommendations_features['Algorithm']
    base_color_palette = sns.color_palette("Set2", n_colors=len(algorithm_names))

    algorithm_color_dict = {}
    class_color_dict = {
        'Baseline': base_color_palette[0],
        'NN': base_color_palette[1],
        'CF': base_color_palette[2],
        'FM': base_color_palette[3],
        'Neural': base_color_palette[5],
    }

    for algorithm in BASELINE_ALGORITHMS:
        algorithm_color_dict[algorithm] = base_color_palette[0]

    for algorithm in NEAREST_NEIGHBOUR_ALGORITHMS:
        algorithm_color_dict[algorithm] = base_color_palette[1]

    for algorithm in COLABORATIVE_FILTERING_ALGORITHMS:
        algorithm_color_dict[algorithm] = base_color_palette[2]

    for algorithm in FACTORISATION_MACHINES_ALGORITHMS:
        algorithm_color_dict[algorithm] = base_color_palette[3]

    for algorithm in NEURAL_ALGORITHMS:
        algorithm_color_dict[algorithm] = base_color_palette[5]

    plot_data = pd.melt(all_algorithm_recommendations_features, id_vars=['Algorithm', 'User', 'Item', 'Position'],
                        value_vars=FEATURES_TO_VISUALISE_NAMES,
                        var_name='Feature', value_name='Value')

    grid = sns.FacetGrid(plot_data, row="Feature", sharey=False, height=2, aspect=3, margin_titles=True,
                         gridspec_kws={"hspace": 0.1})
    grid.axes_dict['#Views'].set_ylim(0, 20000000)
    grid.axes_dict['#Views'].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1000000) + 'M'))

    grid.axes_dict['#Likes'].set_ylim(0, 350000)
    grid.axes_dict['#Likes'].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))

    grid.axes_dict['#Comments'].set_ylim(0, 35000)
    grid.axes_dict['#Comments'].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))

    grid.axes_dict['Duration in Minutes'].set_ylim(0, 120)

    grid.map_dataframe(sns.boxplot, y='Value', x='Algorithm', palette=algorithm_color_dict)
    grid.set_axis_labels("", "")
    grid.set_xticklabels(grid.axes_dict['Duration in Minutes'].get_xticklabels(), rotation=45, horizontalalignment='right',
                         rotation_mode='anchor')
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.subplots_adjust(left=0.1, right=0.94, top=0.98, bottom=0.09)

    # legend plot
    for class_name, color in class_color_dict.items():
        grid.axes[0][0].add_patch(plt.Rectangle((0, 0), 0, 0, color=color, label=class_name))
    grid.axes[0][0].legend(title='Type of algorithm', loc='upper left', bbox_to_anchor=(0.0, 1.1))

    # plt.show()
    plt.savefig("plots/rec_features_zoomed.png")
    plt.clf()

    # 4. Calculate metric values
