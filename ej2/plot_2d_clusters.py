from matplotlib import pyplot as plt


def plot_2d_clusters_scatter(data, labels: list[str], title):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], c=labels)

    # add centroids
    # compute centroids
    centroids = {}
    for i in range(len(labels)):
        if labels[i] not in centroids:
            centroids[labels[i]] = [0, 0]
        centroids[labels[i]][0] += data[i][0]
        centroids[labels[i]][1] += data[i][1]

    for key in centroids:
        centroids[key][0] /= len(labels)
        centroids[key][1] /= len(labels)

    plt.scatter([centroids[key][0] for key in centroids], [
                centroids[key][1] for key in centroids], c='black', marker='x')

    plt.show()
