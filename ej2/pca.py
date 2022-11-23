from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca_to_data(data, n_components=2):
    # standardize data
    scaled_data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    print("Explained variance: ", pca.explained_variance_)

    return pca, pca_data
