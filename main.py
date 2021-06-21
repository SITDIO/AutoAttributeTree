# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.manifold import TSNE
from model import AutoAttributeTree


def plot_tsne(X, clust, title='', savepath='fig.png', random_state=None):
    # TSNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=30,
                n_iter=1000, random_state=random_state)
    data_embedded_tsne = tsne.fit_transform(X)
    n_clust = len(set(list(clust)))

    ax = sns.scatterplot(
        x=data_embedded_tsne[:, 0], y=data_embedded_tsne[:, 1],
        hue=clust,
        palette=sns.color_palette("hls", n_clust),
        legend="full",
        alpha=0.5
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.set_title(f't-SNE {title}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Cluster', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(savepath)
    plt.close()


def main(max_k=10, method='pam', random_state=None):
    df = pd.read_excel(os.path.join('data', 'data.xlsx'))
    data = df.iloc[:, np.r_[1, 4:10]]

    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(data.iloc[:, 1:].values)

    aat = AutoAttributeTree(max_k=max_k, method=method,
                            random_state=random_state)
    aat.fit(data)

    aat.medoids.to_csv(os.path.join('results', 'medoids.csv'), index=False)

    # Niveles complejidad
    niveles = df.iloc[:, 3]
    plot_tsne(X, niveles, title='Complexity levels', savepath=os.path.join(
        'figs', 'complexity_levels.pdf'), random_state=random_state)

    for l in range(1, len(aat.levels2nodes)):
        clusters = aat.get_clusters(level=l)
        clusters = pd.merge(data.iloc[:, 0], clusters, on='Codigo', how='left')
        clusters.to_csv(os.path.join(
            'results', f'clusters_level_{l}.csv'), index=False)

        # Clusters
        clust = clusters.iloc[:, 1]
        plot_tsne(X, clust, title=f'Auto Attribute Tree level {l}', savepath=os.path.join(
            'figs', f'aat_level{l}.pdf'), random_state=random_state)


if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    main(method='pam', random_state=seed)
