# import packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# TASK 1 
# Loading the data 
file_path = 'prog_book.csv'  # Adjust the path as necessary
data = pd.read_csv(file_path)

# Preprocess the data
data['Reviews'] = data['Reviews'].str.replace(',', '').astype(int)  # Convert Reviews to int

# Calculate IQR for each specified column
columns_to_check = ['Rating', 'Reviews', 'Number_Of_Pages', 'Price']
iqr_values = {}

for column in columns_to_check:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_values[column] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}

# Print IQR values
print(iqr_values)

# Plotting box plots for each specified column
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
fig.suptitle('Box Plots to Visualize Outliers', fontsize=16)

sns.boxplot(data=data, y='Rating', ax=axes[0, 0]).set_title('Rating')
sns.boxplot(data=data, y='Reviews', ax=axes[0, 1]).set_title('Reviews')
sns.boxplot(data=data, y='Number_Of_Pages', ax=axes[1, 0]).set_title('Number of Pages')
sns.boxplot(data=data, y='Price', ax=axes[1, 1]).set_title('Price')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the title
plt.show()

# TASK 2

# Multivariate outlier detection

# Standardizing the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[columns_to_check])
data_scaled = pd.DataFrame(data_scaled, columns=columns_to_check)

# Getting all possible pairs of the features
feature_pairs = list(itertools.combinations(columns_to_check, 2))

# Performing DBSCAN for each pair and plotting
for pair in feature_pairs:
    # Applying DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(data_scaled[list(pair)])
    
    # Identifying outliers
    outlier_indices = np.where(clusters == -1)[0]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(data_scaled[pair[0]], data_scaled[pair[1]], c=clusters, cmap='Paired', marker='o', edgecolor='k', s=50)
    plt.title(f'DBSCAN Clustering for {pair[0]} vs. {pair[1]}')
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.colorbar(label='Cluster Label')
    plt.scatter(data_scaled.iloc[outlier_indices][pair[0]], data_scaled.iloc[outlier_indices][pair[1]], color='red', label='Outliers')
    plt.legend()
    plt.show()
    
    # Printing outliers
    if len(outlier_indices) > 0:
        print(f"Outliers for {pair[0]} vs. {pair[1]}:")
        print(data.iloc[outlier_indices][list(pair)])
        print("\n")
    else:
        print(f"No outliers detected for {pair[0]} vs. {pair[1]}.\n")

# Predefined steps: Loading and preprocessing data, scaling, etc.

# Define the feature combinations
feature_combinations = list(combinations(columns_to_check, 3))

for feature_set in feature_combinations:
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(data_scaled[list(feature_set)])
    
    # Identifying outliers
    outlier_indices = np.where(clusters == -1)[0]
    
    # 3D Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_scaled[feature_set[0]], data_scaled[feature_set[1]], data_scaled[feature_set[2]], c=clusters, cmap='Paired', marker='o')
    ax.scatter(data_scaled.iloc[outlier_indices][feature_set[0]], data_scaled.iloc[outlier_indices][feature_set[1]], data_scaled.iloc[outlier_indices][feature_set[2]], color='red', label='Outliers')
    ax.set_title(f'DBSCAN Clustering for {feature_set[0]}, {feature_set[1]}, and {feature_set[2]}')
    ax.set_xlabel(feature_set[0])
    ax.set_ylabel(feature_set[1])
    ax.set_zlabel(feature_set[2])
    ax.legend()
    plt.show()
    
    # Printing outliers
    if len(outlier_indices) > 0:
        print(f"Outliers for {feature_set}:")
        print(data.iloc[outlier_indices][list(feature_set)])
        print("\n")
    else:
        print(f"No outliers detected for {feature_set}.\n")
