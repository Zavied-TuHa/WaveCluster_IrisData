import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Bước 1: Đọc dữ liệu
data = pd.read_csv("D:\Code\Python\KPDL\Clustering_WaveCluster\Iris.csv")

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Bước 2: Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Bước 3: Áp dụng PCA để giảm dữ liệu xuống 2 chiều (nếu dữ liệu có nhiều chiều)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

# Bước 4: Chia dữ liệu thành lưới
grid_size = 16
grid = np.zeros((grid_size, grid_size))

for point in features_2d:
    x_idx = int(point[0] * (grid_size - 1))
    y_idx = int(point[1] * (grid_size - 1))
    grid[x_idx, y_idx] += 1

# Bước 5: Biến đổi sóng Haar
wavelet = 'haar' 
coeffs = pywt.dwt2(grid, wavelet)
cA, (cH, cV, cD) = coeffs 

# Bước 6: Nhận diện các cụm
threshold = 0.1 * np.max(cA)
clusters = np.argwhere(cA > threshold)

cluster_points = []
for cluster in clusters:
    x_center = cluster[0] / (grid_size - 1)
    y_center = cluster[1] / (grid_size - 1)
    cluster_points.append((x_center, y_center))

# Bước 7: Vẽ kết quả
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, label="Data Points")
for point in cluster_points:
    plt.scatter(point[0], point[1], color='red', label="Cluster Center")
plt.title("WaveCluster Result")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
