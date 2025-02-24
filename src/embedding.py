import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from signet.cluster import Cluster
from scipy.sparse import csc_matrix


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
        (
            self.clustered_dfs_train_sets,
            self.clustered_dfs_test_sets,
            self.forecasted_dates,
        ) = self.rolling_embedding_clustering()

    def rolling_embedding_clustering(self):
        (
            train_data_sets,
            test_data_sets,
            forecasted_dates,
        ) = self.data_obj.rolling_train_test_splits
        pool = mp.Pool(mp.cpu_count())
        cluster_labels_sets = pool.map(
            self.spectral_embedding, [data_set for data_set in train_data_sets]
        )
        pool.close()

        clustered_dfs_train_sets = []
        clustered_dfs_test_sets = []
        for i, cluster_labels in enumerate(cluster_labels_sets):
            clustered_dfs_train = {}
            clustered_dfs_test = {}
            for cluster in np.unique(cluster_labels):
                clustered_dfs_train[cluster] = train_data_sets[i].iloc[
                    cluster_labels == cluster
                ]
                clustered_dfs_test[cluster] = test_data_sets[i].iloc[
                    cluster_labels == cluster
                ]
            clustered_dfs_train_sets.append(clustered_dfs_train)
            clustered_dfs_test_sets.append(clustered_dfs_test)

        return clustered_dfs_train_sets, clustered_dfs_test_sets, forecasted_dates

    def spectral_embedding(self, data_set):
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
        correlation_matrix = np.corrcoef(data_set)
        Ap = csc_matrix(
            (correlation_matrix > 0).astype(int)
        )  # Positive adjacency matrix
        An = csc_matrix(
            (correlation_matrix < 0).astype(int)
        )  # Negative adjacency matrix
        k = 2  # cross validate in the future
        cluster = Cluster((Ap, An))
        predicted_clusters = cluster.spectral_cluster_laplacian(
            k=k, normalisation="sym"
        )

        return predicted_clusters


class Sector(Embedding):
    """Sector embedding implementation for financial time series.

    Performs dimensionality reduction using sector embedding followed by
    clustering to identify groups of similarly behaving assets."""

    def __init__(self, run_sett: dict, data_obj):
        super().__init__(run_sett, data_obj)
        self.sector_mapping = (
            pd.read_csv(
                self._run_sett["input_dir_mappings"] + "sector_mapping_sp1500.csv"
            )
            .iloc[:, 1:3]
            .rename(columns={"SPY.1": "ticker", "0": "sector"})
            .set_index("ticker")
        )
        (
            self.clustered_dfs_train_sets,
            self.clustered_dfs_test_sets,
            self.forecasted_dates,
        ) = self.rolling_sector_clustering()

    def rolling_sector_clustering(self):
        (
            train_data_sets,
            test_data_sets,
            forecasted_dates,
        ) = self.data_obj.rolling_train_test_splits

        clustered_dfs_train_sets = []
        clustered_dfs_test_sets = []

        sectors = self.sector_mapping.sector.unique()

        for i, (train_data_set, test_data_set) in enumerate(
            zip(train_data_sets, test_data_sets)
        ):
            clustered_dfs_train = {}
            clustered_dfs_test = {}
            train_data_set = pd.merge(
                train_data_set,
                self.sector_mapping,
                left_index=True,
                right_index=True,
                how="inner",
            )
            test_data_set = pd.merge(
                test_data_set,
                self.sector_mapping,
                left_index=True,
                right_index=True,
                how="inner",
            )
            for sector in np.unique(sectors):
                clustered_dfs_train[sector] = train_data_set.loc[
                    train_data_set.sector == sector
                ].drop(columns="sector")
                clustered_dfs_test[sector] = test_data_set.loc[
                    test_data_set.sector == sector
                ].drop(columns="sector")
            clustered_dfs_train_sets.append(clustered_dfs_train)
            clustered_dfs_test_sets.append(clustered_dfs_test)

        return clustered_dfs_train_sets, clustered_dfs_test_sets, forecasted_dates


# first setup of autoencoder, specifics are not correct have to look into how to do it correctly
class Encoder(nn.Module):
    """Simple encoder network that can be extended later.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    encoding_dim : int
        Dimension of the encoded representation
    """

    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Simple decoder network that can be extended later.

    Parameters
    ----------
    encoding_dim : int
        Dimension of the encoded representation
    output_dim : int
        Dimension of output features
    """

    def __init__(self, encoding_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(Embedding):
    """Autoencoder implementation for financial time series.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing runtime settings and parameters
    data_obj : object
        Object containing the financial data to be embedded

    Attributes
    ----------
    encoder : torch.nn.Module
        Neural network for encoding data
    decoder : torch.nn.Module
        Neural network for decoding data
    encoding_dim : int
        Dimension of the encoded representation
    device : torch.device
        Device to run computations on (CPU/GPU)
    """

    def __init__(self, run_sett: dict, data_obj):
        super().__init__(run_sett, data_obj)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.encoding_dim = run_sett["embeddings"]["autoencoder"]["encoding_dim"]

        # Initialize data
        self._correlation_matrix = np.corrcoef(self.data_obj.data[:20])

        # Setup model
        self.input_dim = self._correlation_matrix.shape[1]
        self.encoder = Encoder(self.input_dim, self.encoding_dim).to(self.device)
        self.decoder = Decoder(self.encoding_dim, self.input_dim).to(self.device)

        # Train autoencoder and get embeddings
        self.encoded_data = self._train_autoencoder()
        self.cluster_labels = self.cluster_embedding(self.encoded_data)
        self.clustered_dfs = self.clustered_data()

    def _train_autoencoder(self, batch_size=32, epochs=100, learning_rate=0.001):
        """Train the autoencoder.

        Parameters
        ----------
        batch_size : int, optional
            Size of training batches
        epochs : int, optional
            Number of training epochs
        learning_rate : float, optional
            Learning rate for optimization

        Returns
        -------
        numpy.ndarray
            Encoded representation of the input data
        """
        # Prepare data
        data = torch.FloatTensor(self._correlation_matrix).to(self.device)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                # Get batch
                x = batch[0]

                # Forward pass
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)

                # Compute loss
                loss = criterion(decoded, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}"
                )

        # Get final embeddings
        with torch.no_grad():
            encoded_data = self.encoder(data).cpu().numpy()

        return encoded_data

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
        """
        return super().cluster_embedding(embedding)

    def clustered_data(self):
        """Group the original data by cluster assignments."""
        return super().clustered_data()
