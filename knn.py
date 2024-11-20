# Importing libraries
import numpy as np  
from collections import Counter

# Sample dataset: [Weight, Color], Class
X_train = np.array([[150, 1], [130, 1], [160, 2], [140, 2], [170, 1], [120, 1]])  

y_train = np.array(['Apple', 'Apple', 'Pear', 'Pear', 'Apple', 'Apple'])  

# Euclidean distance function
def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points.
    This function measures the 'straight-line' distance between two feature vectors in n-dimensional space."""
    return np.sqrt(np.sum((point1 - point2)**2))

# KNN algorithm
def knn(X_train, y_train, query_point, k=3):
    """K-Nearest Neighbors algorithm to classify a query point.
    Parameters:
    - X_train: Training data (features)
    - y_train: Training data (labels)
    - query_point: Data point to classify
    - k: Number of nearest neighbors to consider
    """
    
    # Step 1: Initialize an empty list to store distances and corresponding labels
    distances = []  # This will hold tuples of (distance, class label)

    # Step 2: Calculate distances between query point and all training data points
    for i in range(len(X_train)):  
        # Iterate over each point in the training data
        distance = euclidean_distance(X_train[i], query_point)  
        # Calculate the distance between the query point and the current training point
        distances.append((distance, y_train[i]))  
        # Append a tuple (distance, label) to the distances list

    # Step 3: Sort the distances (first element of each tuple)
    distances.sort(key=lambda x: x[0])

    # Step 4: Get the classes of the k nearest neighbors
    k_nearest_classes = [distances[i][1] for i in range(k)]  

    # Step 5: Return the most common class (majority vote)
    most_common_class = Counter(k_nearest_classes).most_common(1)[0]
    return most_common_class

# === An example usage ====
query_point = np.array([145, 1])  # Weight=145g, Color=1 (red)
k = 3

# Predict the class for the new point
prediction = knn(X_train, y_train, query_point, k)
print(f"The predicted class for the query point is: {prediction}")