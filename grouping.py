import os
import shutil
import glob
import numpy as np
from sklearn.cluster import BisectingKMeans
from PIL import Image

# directories
data_dir = r"C:\Users\Aqua SD\Dropbox\AquaSD - Docs\Aqua SD - POST & DONE\Post to Site\POST - Retail Website\000 - NAME AND SORT"
end_loc = r"C:\Users\Aqua SD\Dropbox\AquaSD - Docs\Aqua SD - POST & DONE\Post to Site\POST - Retail Website\000 - NAME AND SORT\cluster test"

# Create a list of files to be clustered
file_list = glob.glob(os.path.join(data_dir, "*"))

# Create features to cluster files based on
features = []
for file_path in file_list:
    image = Image.open(file_path)
    features.append(np.array(image).flatten())

num_clusters = 10  # Adjust the number of clusters
kmeans = BisectingKMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Create folders for each cluster
os.makedirs(end_loc, exist_ok=True)
cluster_folders = {}
for i in range(num_clusters):
    cluster_folder = os.path.join(end_loc, f"cluster_{i}")
    os.makedirs(cluster_folder, exist_ok=True)
    cluster_folders[i] = cluster_folder

# Move Into Cluster Folders
for i, file_path in enumerate(file_list):
    cluster_index = kmeans.labels_[i]
    cluster_folder = cluster_folders[cluster_index]
    # Using copy instead of move since we are still testing if this will work long term
    shutil.copy(file_path, os.path.join(cluster_folder, os.path.basename(file_path)))
