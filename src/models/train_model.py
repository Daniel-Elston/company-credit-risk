from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

from config import ModelState
from utils.setup_env import setup_project_env

project_dir, config, setup_logs = setup_project_env()


class ClusteringTool:
    def __init__(self):
        self.ms = ModelState()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fig_path = Path(config["path"]["exploration"])
        self.cluster_path = Path(config["path"]["clustering"])

    def fit_model(self, df, method, **kwargs):
        """Fit clustering model based on method"""
        if method == 'kmeans':
            model = KMeans(**kwargs)
        elif method == 'dbscan':
            model = DBSCAN(**kwargs)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(**kwargs)
        else:
            raise ValueError("Invalid clustering method")

        labels = model.fit_predict(df)
        self.logger.info(f"{method} clustering complete.")
        return model, labels

    def evaluate_clustering(self, df, labels):
        """Evaluate clustering performance using silhouette score and Davies-Bouldin index."""
        silhouette_avg = silhouette_score(df, labels)  # Above 0.5 is good, 1 is best
        db_index = davies_bouldin_score(df, labels)  # Below 1 is good, 0 is best
        self.logger.info(f"Silhouette Score: {round(silhouette_avg, 2)}, Davies-Bouldin Index: {round(db_index, 2)}")
        return silhouette_avg, db_index

    def plot_clusters(self, df, labels, title):
        """Plot clustering results."""
        df['Cluster'] = labels
        print(df['Cluster'].value_counts())
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='PC1', y='PC2', hue='Cluster',
            palette=sns.color_palette('hls', n_colors=len(df['Cluster'].unique())),
            alpha=0.4, s=5)
        plt.title(title)
        plt.savefig(self.fig_path / f'{title}.png')
        plt.show()
        plt.close()

    def save_metrics(self, metrics, run_number):
        """Save clustering evaluation metrics to JSON."""
        with open(self.cluster_path / f'clustering_metrics_run_{run_number}.json', 'w') as f:
            json.dump(metrics, f)
        self.logger.info(f"Clustering metrics saved for run {run_number}.")

    def pipeline(self, df, run_number, **clustering_params):
        """Pipeline to run multiple clustering analyses."""
        metrics = {}

        for method in self.ms.clustering_methods:
            model, labels = self.fit_model(df, method=method, **clustering_params.get(method, {}))
            silhouette_avg, db_index = self.evaluate_clustering(df, labels)
            metrics[method] = {
                "silhouette_score": silhouette_avg,
                "davies_bouldin_index": db_index
            }
            self.plot_clusters(df, labels, f'{method}_clustering_run_{run_number}')

        self.save_metrics(metrics, run_number)
        self.logger.info(f"Clustering pipeline completed for run {run_number}.")
