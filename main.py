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


def main():
    df = pd.read_excel(os.path.join('data', 'data.xlsx'))
    data = df.iloc[:, np.r_[1, 4:10]]

    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(data.iloc[:, 1:].values)

    aat = AutoAttributeTree()
    aat.fit(data)

    # TSNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
    data_embedded_tsne = tsne.fit_transform(X)
    # Niveles complejidad
    niveles = df.iloc[:, 3]
    n_niveles = len(set(list(niveles)))

    ax = sns.scatterplot(
        x=data_embedded_tsne[:, 0], y=data_embedded_tsne[:, 1],
        hue=niveles,
        palette=sns.color_palette("hls", n_niveles),
        legend="full",
        alpha=0.5
    )
    ax.set_title(f't-SNE Complexity Levels')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.savefig(os.path.join('figs', 'complexity_levels.png'))
    plt.close()

    for l in range(1, 5):
        clusters = aat.get_clusters(level=l)
        clusters = pd.merge(data.iloc[:, 0], clusters, on='Codigo', how='left')

        # Clusters
        clust = clusters.iloc[:, 1]
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
        ax.set_title(f't-SNE Auto Attribute Tree level {l}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend(title='Cluster', loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join('figs', f'aat_level{l}.png'))
        plt.close()


if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    main()
