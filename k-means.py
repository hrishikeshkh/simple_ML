import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

X = [[0.180, 1.786], [0.353, 1.240], [0.940, 1.566], [1.486, 0.759], [1.266, 1.106], [1.540, 0.419], [0.459, 1.799], [0.773, 0.186]]

df = pd.DataFrame(X)

# Given data
X = [[0.180, 1.786], [0.353, 1.240], [0.940, 1.566], [1.486, 0.759], [1.266, 1.106], [1.540, 0.419], [0.459, 1.799], [0.773, 0.186]]

# Convert X to a DataFrame
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# Perform clustering using KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Print the DataFrame with cluster assignments
print(df)

#make a prediction on a new sample = [0.606, 0.906]
new_sample = [[0.606, 0.906]]
print(kmeans.predict(new_sample))