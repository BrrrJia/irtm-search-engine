import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io
import os


def k_means(data, k=2, seed=42, max_iter=100, tolerance=1e-4):
    np.random.seed(seed)
    data = data.toarray()

    # compute the distance and classify
    classes = np.zeros(data.shape[0], dtype=int)
    done = False
    step = 0
    total_rss = float("inf")
    first = True
    while not done and step <= max_iter:
        if first:
            # initialise random centroids
            centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
            first = False
        else:
            # compute means to get new centroids
            for i in range(k):
                cluster_points = data[classes == i]
                if cluster_points.shape[0] > 0:  # only update non-empty clusters
                    centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    centroids[i] = data[np.random.randint(data.shape[0])]
        # compute distance and classify
        for idx, d in enumerate(data):
            min_dis = float("inf")
            best_cls = None
            for i, c in enumerate(centroids):
                distance = np.linalg.norm(d - c)
                if distance <= min_dis:
                    min_dis = distance
                    best_cls = i
                    classes[idx] = best_cls
        # classes = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)

        # compute rss
        rss = [
            np.sum(np.linalg.norm((data[classes == i] - centroids[i]) ** 2, axis=1))
            for i in range(k)
        ]

        old_total_rss = total_rss
        total_rss = sum(rss)
        if abs(old_total_rss - total_rss) < tolerance:
            done = True
        step += 1
    # print(step)
    return classes, centroids, total_rss, np.array(rss)


def optimal_k_means(data, k=2, n_init=50):
    best_result = None
    best_rss = float("inf")
    for seed in range(n_init):
        classes, centroids, total_rss, rss = k_means(data, k, seed=seed)
        if total_rss < best_rss:
            best_result = (classes, centroids, total_rss, np.array(rss))
    return best_result


def find_closest_docs_to_centroids(data, classes, centroids):
    data = data.toarray()
    closest_doc_indices = []

    for i in range(centroids.shape[0]):
        cluster_points = data[classes == i]
        cluster_indices = np.where(classes == i)[0]

        if cluster_points.shape[0] == 0:
            closest_doc_indices.append(None)
            continue

        # compute the distance between cluster points to the centroids
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        closest_index_in_cluster = np.argmin(distances)
        closest_doc_index = cluster_indices[closest_index_in_cluster]

        closest_doc_indices.append(closest_doc_index)

    return closest_doc_indices


def plot_clusters(
    data,
    labels,
    centroids=None,
    title="K-means Clustering",
    highlight=None,
    save_to_file=True,
):
    # Reduce data to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Transform centroids if available
    reduced_centroids = pca.transform(centroids) if centroids is not None else None

    fig = plt.figure(figsize=(10, 7))

    # Plot each cluster in a different color
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        plt.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
        )

    # Plot centroids
    if reduced_centroids is not None:
        plt.scatter(
            reduced_centroids[:, 0],
            reduced_centroids[:, 1],
            c="black",
            marker="X",
            s=100,
            label="Centroids",
        )

    # Highlight a specific cluster (e.g. with minimum rss)
    if highlight is not None:
        plt.scatter(
            reduced_centroids[highlight, 0],
            reduced_centroids[highlight, 1],
            c="red",
            marker="*",
            s=50,
            label=f"Small RSS Cluster {highlight}",
        )

    plt.title(title, fontsize=14)
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tick_params(labelsize=10)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    plt.tight_layout()

    if save_to_file:
        filepath = "output/k-means.png"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf
