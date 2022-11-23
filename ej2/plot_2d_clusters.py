from matplotlib import pyplot as plt


def plot_2d_clusters_scatter(x_data, y_data, labels: list[str], title, x_label, y_label):
    plt.figure()
    plt.title(title)
    plt.scatter(x_data, y_data, c=labels, cmap="rainbow", marker="o")
    plt.legend(
        labels,
        loc="upper right",
        bbox_to_anchor=(1.3, 1),
        ncol=1,
        title="Clusters",
        fancybox=True,
        shadow=True,
    )

    # add centroids
    # compute centroids
    centroids = {}
    for i in range(len(labels)):
        if labels[i] not in centroids:
            centroids[labels[i]] = [0, 0]
        centroids[labels[i]][0] += x_data[i]
        centroids[labels[i]][1] += y_data[i]

    for key in centroids:
        label_count = labels.count(key)
        centroids[key][0] /= label_count
        centroids[key][1] /= label_count

    plt.scatter([centroids[key][0] for key in centroids], [
                centroids[key][1] for key in centroids], c='blue', marker='x')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
