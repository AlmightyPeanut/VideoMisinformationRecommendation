import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS = {
    'MostPop': {
        'nDCG': 0.10052832322147234,
        'Precision': 0.1,
        'Recall': 0.07412119313435103,
        'MRR': 0.22150793650793651,
        'NS': 0.11000000000000001,
        'SERP-MS': 0.13454545454545452
    },
    'Random': {
        'nDCG': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'MRR': 0.0,
        'NS': 0.02,
        'SERP-MS': 0.01818181818181818
    },
    'I-k NN': {
        'nDCG': 0.9529918645140908,
        'Precision': 0.93,
        'Recall': 0.7063382889040783,
        'MRR': 1.0,
        'NS': 0.21000000000000002,
        'SERP-MS': 0.17272727272727273
    },
    'U-k NN': {
        'nDCG': 0.9618263485602354,
        'Precision': 0.9400000000000001,
        'Recall': 0.7125882889040784,
        'MRR': 1.0,
        'NS': 0.22000000000000003,
        'SERP-MS': 0.18
    },
    'MF': {
        'nDCG': 0.0931646189676573,
        'Precision': 0.06,
        'Recall': 0.04157268170426065,
        'MRR': 0.37222222222222223,
        'NS': 0.03,
        'SERP-MS': -0.0018181818181818188
    },
    'NeuMF': {
        'nDCG': 0.10192295090658754,
        'Precision': 0.05,
        'Recall': 0.03768740031897926,
        'MRR': 0.45,
        'NS': 0.030000000000000006,
        'SERP-MS': 0.03272727272727273
    },
    'DMF': {
        'nDCG': 0.9624330134579268,
        'Precision': 0.95,
        'Recall': 0.7197311460469356,
        'MRR': 1.0,
        'NS': 0.18,
        'SERP-MS': 0.18
    },
    'BPRMF': {
        'nDCG': 0.11970405297956395,
        'Precision': 0.11000000000000001,
        'Recall': 0.07513471177944861,
        'MRR': 0.3183333333333333,
        'NS': 0.08, 'SERP-MS': 0.06
    },
    'CML': {
        'nDCG': 0.09189485840123816,
        'Precision': 0.07,
        'Recall': 0.05565998329156223,
        'MRR': 0.3083333333333333,
        'NS': 0.09000000000000001,
        'SERP-MS': 0.06909090909090909
    },
    'NNMF': {
        'nDCG': 0.26189771620998503,
        'Precision': 0.22999999999999998,
        'Recall': 0.14469924812030074,
        'MRR': 0.37,
        'NS': -0.01,
        'SERP-MS': -0.00909090909090909
    },
    'PMF': {
        'nDCG': 0.1388066276259041,
        'Precision': 0.12,
        'Recall': 0.08932159945317839,
        'MRR': 0.3401190476190476,
        'NS': 0.01,
        'SERP-MS': 0.010909090909090908
    },
    'GMF': {
        'nDCG': 0.12657800645134615,
        'Precision': 0.09,
        'Recall': 0.05965852130325814,
        'MRR': 0.4541666666666666,
        'NS': 0.08,
        'SERP-MS': 0.07272727272727272
    },
    'LogMF': {
        'nDCG': 0.006943122193677725,
        'Precision': 0.01,
        'Recall': 0.008333333333333333,
        'MRR': 0.0125,
        'NS': 0.0,
        'SERP-MS': 0.0
    },
}

RESULTS_INDEX_MAPPING = {
    'GeneralizedMF': 'GMF',
    'SVDpp': 'SVD++',
    'BPRMF': 'BPRMF',
    'ItemKNN': 'I-k NN',
    'MostPop': 'MostPop',
    'FFM': 'FFM',
    'Random': 'Random',
    'NonNegMF': 'NNMF',
    'MF': 'MF',
    'PMF': 'PMF',
    'CML': 'CML',
    'FunkSVD': 'FunkSVD',
    'UserKNN': 'U-k NN',
    'NeuMF': 'NeuMF',
    'LogisticMatrixFactorization': 'LogMF',
    'DMF': 'DMF',
}

BASELINE_ALGORITHMS = ['Random', 'MostPop']
NEAREST_NEIGHBOUR_ALGORITHMS = ['I-k NN', 'U-k NN']
COLABORATIVE_FILTERING_ALGORITHMS = ['BPRMF', 'CML', 'FunkSVD', 'LogMF', 'MF', 'NNMF', 'PMF', 'SVD++']
FACTORISATION_MACHINES_ALGORITHMS = ['DeepFM', 'FFM', 'NFM']
NEURAL_ALGORITHMS = ['ConvMF', 'DMF', 'GMF', 'MultiVAE', 'NeuMF']


if __name__ == '__main__':
    # Performance metrics

    # results_df = pd.DataFrame.from_dict(RESULTS).T
    results_df = pd.read_csv('results/yaudit_non_hybrid/performance/rec_cutoff_10_relthreshold_1_2023_08_08_20_51_52.tsv', sep='\t', index_col=0)
    results_df.index = results_df.index.map(lambda model_name: model_name.split('_')[0])
    results_df.index = results_df.index.map(RESULTS_INDEX_MAPPING.get)
    print(results_df.style.format(precision=3).to_latex(
        column_format='r' + '|c' * results_df.shape[1]))

    # color palette per type of algorithm
    algorithm_names = results_df.index
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

    results_df = results_df.sort_values(by='nDCG', ascending=False)
    for metric in results_df.columns:
        metric_data = results_df[metric].reset_index()
        metric_data.columns = ['Algorithm', 'Value']
        # metric_data.sort_values(by='Value', inplace=True, ascending=False)
        plot = sns.barplot(metric_data, x='Algorithm', y='Value', palette=algorithm_color_dict)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right', rotation_mode='anchor')
        plot.set_xlabel("")
        plot.set_ylabel("")

        # legend plot
        for class_name, color in class_color_dict.items():
            plot.add_patch(plt.Rectangle((0, 0), 0, 0, color=color, label=class_name))
        plot.legend(frameon=False, title='Type of algorithm')

        # plotting and saving the plots
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"plots/{metric}.png")
        plt.clf()

    statistical_significane_df = pd.read_csv('results/yaudit_non_hybrid/performance/stat_paired_ttest_cutoff_10_relthreshold_1_2023_08_08_20_51_52.tsv', sep='\t', header=None)
    statistical_significane_df.columns = ['Alg1', 'Alg2', 'Metric', 'Value']
    statistical_significane_df['Alg1'] = statistical_significane_df['Alg1'].map(lambda model_name: model_name.split('_')[0])
    statistical_significane_df['Alg1'] = statistical_significane_df['Alg1'].map(RESULTS_INDEX_MAPPING.get)
    statistical_significane_df['Alg2'] = statistical_significane_df['Alg2'].map(lambda model_name: model_name.split('_')[0])
    statistical_significane_df['Alg2'] = statistical_significane_df['Alg2'].map(RESULTS_INDEX_MAPPING.get)

    for metric in statistical_significane_df['Metric'].unique():
        metric_stat_df = statistical_significane_df[statistical_significane_df['Metric'] == metric]
        metric_stat_df = metric_stat_df.drop('Metric', axis=1).set_index(['Alg1', 'Alg2']).unstack()
        metric_stat_df.columns = metric_stat_df.columns.droplevel()
        metric_stat_df.sort_index(axis=0, key=lambda x: x.map(lambda e: results_df.index.get_loc(e)), inplace=True)
        metric_stat_df.sort_index(axis=1, key=lambda x: x.map(lambda e: results_df.index.get_loc(e)), inplace=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = sns.color_palette('blend:#66c2a5,#ffd92f', as_cmap=True)
        cmap.set_over('#e02c2c')
        plot = sns.heatmap(metric_stat_df.fillna(1.0), vmin=.0, center=0.04, vmax=0.05, cmap=cmap, ax=ax)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right', rotation_mode='anchor')
        plot.set_xlabel("")
        plot.set_ylabel("")
        # plt.title(metric)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plots/stat_significance_{metric}.png')
        plt.clf()
