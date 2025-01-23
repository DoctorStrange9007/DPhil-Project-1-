import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans


class Embedding:
    """Base class for dimensionality reduction techniques.

    This class provides a common interface for different embedding methods
    used to reduce the dimensionality of financial time series data.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing runtime settings and parameters
    data_obj : object
        Object containing the financial data to be embedded
        Must have a 'data' attribute containing a pandas DataFrame

    Notes
    -----
    Currently supported embedding methods:
    - Spectral embedding
    - Auto encoder (planned)
    """

    def __init__(self, run_sett: dict, data_obj):
        self.data_obj = data_obj
        self._run_sett = run_sett


class Spectral(Embedding):
    """Spectral embedding implementation for financial time series.

    Performs dimensionality reduction using spectral embedding followed by
    clustering to identify groups of similarly behaving assets.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing runtime settings and parameters
    data_obj : object
        Object containing the financial data to be embedded

    Attributes
    ----------
    correlation_matrix : numpy.ndarray
        Correlation matrix of the input data
    cluster_labels : numpy.ndarray
        Cluster assignments for each asset
    clustered_dfs : dict
        Dictionary of DataFrames containing data grouped by cluster

    Notes
    -----
    Currently processes only the first 20 assets due to computational constraints
    """

    def __init__(self, run_sett: dict, data_obj):
        super().__init__(run_sett, data_obj)
        self._correlation_matrix = np.corrcoef(
            self.data_obj.data[:20]
        )  # for now due to computational time)
        self.cluster_labels = self.cluster_embedding(
            self.spectral_embedding(self._correlation_matrix)
        )
        self.clustered_dfs = self.clustered_data()

    def spectral_embedding(self, correlation_matrix):
        """Apply spectral embedding to the correlation matrix.

        Parameters
        ----------
        correlation_matrix : numpy.ndarray
            Square matrix of pairwise correlations between assets

        Returns
        -------
        numpy.ndarray
            Lower-dimensional embedding of the input data (n_samples, n_components)

        Notes
        -----
        Uses precomputed affinity matrix with 10 components for embedding
        """
        spectral = SpectralEmbedding(n_components=10, affinity="precomputed")
        spectral_embedding = spectral.fit_transform(correlation_matrix)

        return spectral_embedding

    def cluster_embedding(self, embedding):
        """Cluster the embedded data using K-means.

        Parameters
        ----------
        embedding : numpy.ndarray
            Lower-dimensional embedding of the data

        Returns
        -------
        numpy.ndarray
            Cluster labels for each asset

        Notes
        -----
        Currently uses fixed k=2 clusters
        Future implementation will use cross-validation for optimal k
        """
        k = 2  # cross validate in the future
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=1)
        kmeans.fit(embedding)
        cluster_labels = kmeans.labels_

        return cluster_labels

    def clustered_data(self):
        """Group the original data by cluster assignments.

        Returns
        -------
        dict
            Dictionary where keys are cluster labels and values are
            pandas DataFrames containing the data for assets in that cluster

        Notes
        -----
        Currently only processes first 20 assets due to computational constraints
        """
        clustered_dfs = {}
        for cluster in np.unique(self.cluster_labels):
            clustered_dfs[cluster] = self.data_obj.data[:20].iloc[
                self.cluster_labels == cluster
            ]  # for now due to computational time)

        return clustered_dfs
