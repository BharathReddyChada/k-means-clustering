import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(filename):
    # Load the dataset from an Excel file
    return pd.read_excel(filename)

def initialize_centroids(X, k):
    # Randomly initialize k centroids from the dataset
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]
    return centroids

def closest_centroid(X, centroids):
    # Calculate the distance of each point in X to each centroid
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    # Assign each point to the closest centroid
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    # Update centroids by computing the mean of points assigned to each centroid
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(X, k, max_iters=100):
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        # Assign points to the closest centroid
        labels = closest_centroid(X, centroids)
        # Update centroids
        new_centroids = update_centroids(X, labels, k)
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Load dataset
filename = 'Mall_Customers copy.xlsx'  # Update this with the path to your Excel file
df = load_dataset(filename)

# Assuming the dataset has two features 'Feature1' and 'Feature2'
X = df[['Annual Income (k$)', 'Age']].values

# Number of clusters
k = 5


# Run k-means clustering
centroids, labels = k_means(X, k)

# Plotting the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.title('K-means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Age')
plt.show()
