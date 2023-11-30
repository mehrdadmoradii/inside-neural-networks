import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_squares_dataset(n_samples=1000, noise_level=0.1):
    num_samples_per_square = n_samples // 4
    centers = [[0, 3.75], [0, 0], [3.5, 0], [3.5, 3.75]]
    variance = [[noise_level, 0], [0, noise_level]]
    X = np.vstack([
        np.random.multivariate_normal(center, variance, num_samples_per_square) for center in centers
    ])
    y = np.hstack([
        np.full(num_samples_per_square, i % 2) for i in range(4)
    ])
    return X, y


def standardize_dataset(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def split_dataset(X, y):
    # X_train, X_test, y_train, y_test =
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_dataset(n_samples=2000, noise_level=0.2):
    X, y = get_squares_dataset(n_samples=n_samples, noise_level=noise_level)
    X, y = standardize_dataset(X, y)
    return split_dataset(X, y)
