import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

# -------------------------------------------------------------------
# 1. Define all functions at the top level.
# -------------------------------------------------------------------

def initData(filename, training_ratio, validation_ratio, test_ratio, attributes):
    """Load data from CSV, split into training, validation, test."""
    with open(filename, 'r', newline='') as file:
        file_csv = csv.reader(file)
        full_header = next(file_csv)  # Read the full header row

        # Identify indices for the chosen attributes
        chosen_indices = [full_header.index(attr) for attr in attributes]
        labels = []
        data = []

        for row in file_csv:
            # Skip rows that don't match header length
            if len(row) < len(full_header):
                continue

            # Extract chosen attributes and convert to float
            features = [float(row[i]) for i in chosen_indices]

            # Assume the last column is the label
            label = row[-1]
            data.append(features + [label])

            if label not in labels:
                labels.append(label)

        len_data = len(data)
        print(f"Total rows read: {len_data}")

        # Split into training, validation, and test sets
        train_idx = int(training_ratio * len_data)
        val_idx = int((training_ratio + validation_ratio) * len_data)
        
        training_data = data[:train_idx]
        validation_data = data[train_idx:val_idx]
        test_data = data[val_idx:]

        if len(training_data) == 0 or len(validation_data) == 0:
            print("[ERROR] Training or validation set is empty. Check your data split.")
            exit()
    
    chosen_header = [full_header[i] for i in chosen_indices]
    return chosen_header, training_data, validation_data, test_data, labels

def minkowski_distance(point_1, point_2, p):
    return sum(abs(a - b) ** p for a, b in zip(point_1, point_2)) ** (1 / p)

def knn(training_data, new_point, k, p):
    """Find the majority label among the k nearest neighbors."""
    # Compute distances from new_point to every row in training_data
    distances = [
        (minkowski_distance(new_point, row[:-1], p), row[-1])
        for row in training_data
    ]
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Vote among the k nearest
    votes = {}
    for i in range(k):
        label = distances[i][1]
        votes[label] = votes.get(label, 0) + 1
    
    # Return label with max votes
    return max(votes, key=votes.get)

def compute_accuracy(training_data, data, k, p):
    """Compute accuracy on given data using KNN with specified k and p."""
    if not data:
        return 0.0
    correct = 0
    for point in data:
        predicted_label = knn(training_data, point[:-1], k, p)
        actual_label = point[-1]
        if predicted_label == actual_label:
            correct += 1
    return correct / len(data)



def compute_accuracies_for_k_p(args):
    k, p, training_data, validation_data, test_data = args
    val_acc = compute_accuracy(training_data, validation_data, k, p)
    test_acc = compute_accuracy(training_data, test_data, k, p)
    return k, p, val_acc, test_acc

from sklearn.preprocessing import StandardScaler

def scale_data(training_data, validation_data, test_data):
    """
    Scale features using StandardScaler while preserving labels
    """
    # Extract features and labels
    train_features = np.array([row[:-1] for row in training_data])
    val_features = np.array([row[:-1] for row in validation_data])
    test_features = np.array([row[:-1] for row in test_data])
    
    # Get labels
    train_labels = [row[-1] for row in training_data]
    val_labels = [row[-1] for row in validation_data]
    test_labels = [row[-1] for row in test_data]
    
    # Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    
    # Combine scaled features with labels
    training_scaled = [list(features) + [label] for features, label in zip(train_scaled, train_labels)]
    validation_scaled = [list(features) + [label] for features, label in zip(val_scaled, val_labels)]
    test_scaled = [list(features) + [label] for features, label in zip(test_scaled, test_labels)]
    
    return training_scaled, validation_scaled, test_scaled

dataset_file = 'final_pollution_dataset.csv'
attributes = ['NO2','CO', 'Proximity_to_Industrial_Areas'] 
k = 15
p = 3
grid_step = 0.05  # Grid step size for faster computation

# Split ratios
training_ratio = 0.7
validation_ratio = 0.1
test_ratio = 0.2

def main():
    # Load and prepare data
    chosen_header, training_data, validation_data, test_data, labels = initData(
        filename=dataset_file,
        training_ratio=training_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        attributes=attributes
    )
    training_data, validation_data, test_data = scale_data(training_data, validation_data, test_data)

    # Setup parameters
    k_values = range(1, 20)
    p_values = range(1, 10)
    
    # Initialize accuracy matrices
    val_accuracies = np.zeros((len(k_values), len(p_values)))
    test_accuracies = np.zeros((len(k_values), len(p_values)))
    
    # Prepare arguments for parallel processing
    args_list = [
        (k, p, training_data, validation_data, test_data)
        for k in k_values
        for p in p_values
    ]
    
    # Compute accuracies in parallel
    print(f"Starting parallel computation for {len(args_list)} (k, p) combinations...")
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(compute_accuracies_for_k_p, args_list))
        
    # Process results
    for k, p, val_acc, test_acc in results:
        i = k - 1  # Adjust for 0-based indexing
        j = p - 1
        val_accuracies[i][j] = val_acc
        test_accuracies[i][j] = test_acc
        
    # Print results
    print("\nFinal Results:")
    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            print(f"k={k}, p={p} -> val_acc={val_accuracies[i][j]:.3f}, "
                  f"test_acc={test_accuracies[i][j]:.3f}")
    
    # Plot results
    plot_heatmaps(val_accuracies, test_accuracies, k_values, p_values)

def plot_heatmaps(val_accuracies, test_accuracies, k_values, p_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Validation accuracy heatmap
    sns.heatmap(val_accuracies, annot=True, fmt='.3f',
                xticklabels=p_values, yticklabels=k_values,
                ax=ax1, cmap='viridis')
    ax1.set_title('Validation Accuracy')
    ax1.set_xlabel('p value')
    ax1.set_ylabel('k value')
    
    # Test accuracy heatmap
    sns.heatmap(test_accuracies, annot=True, fmt='.3f',
                xticklabels=p_values, yticklabels=k_values,
                ax=ax2, cmap='viridis')
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('p value')
    ax2.set_ylabel('k value')
    
    plt.tight_layout()
    plt.savefig('k_p_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()