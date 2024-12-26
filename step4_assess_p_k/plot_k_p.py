import sns
import numpy as np
import matplotlib.pyplot as plt
import csv

# Initialize the data
def initData(filename, training_ratio, validation_ratio, test_ratio, attributes):
    with open(filename, 'r', newline='') as file:
        file_csv = csv.reader(file)
        full_header = next(file_csv)  # Read the full header row

        # Identify indices for the chosen attributes
        chosen_indices = [full_header.index(attr) for attr in attributes]
        labels = []
        data = []

        for row in file_csv:
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


# %%
def minkowski_distance(point_1, point_2, p):
    return sum(abs(a - b)**p for a, b in zip(point_1, point_2))**(1/p)


# %%
def knn(training_data, new_point, k, p):
    distances = [
        (minkowski_distance(new_point, row[:-1], p), row[-1])
        for row in training_data
    ]
    distances.sort(key=lambda x: x[0])
    
    # Perform voting among the k nearest neighbors
    votes = {}
    for i in range(k):
        label = distances[i][1]
        votes[label] = votes.get(label, 0) + 1
    
    return max(votes, key=votes.get)

# %%
def compute_accuracy(training_data, data, k, p):

    if not data:
        return 0
    correct = sum(
        knn(training_data, point[:-1], k, p) == point[-1]
        for point in data
    )
    return correct / len(data)

# %%
dataset_file = 'final_pollution_dataset.csv'
attributes = ['SO2', 'CO', 'Proximity_to_Industrial_Areas']  # Two selected attributes

# Split ratios
training_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

# Load data
chosen_header, training_data, validation_data, test_data, labels = initData(
    filename=dataset_file,
    training_ratio=training_ratio,
    validation_ratio=validation_ratio,
    test_ratio=test_ratio,
    attributes=attributes
)

# Initialize 2D arrays with zeros
k_range = range(1, 10)
p_range = range(1, 6)
val_accuracies = [[0 for _ in p_range] for _ in k_range]
test_accuracies = [[0 for _ in p_range] for _ in k_range]

# Compute accuracies
for i, k in enumerate(k_range):
    for j, p in enumerate(p_range):
        acc_val = compute_accuracy(training_data, validation_data, k, p)
        acc_test = compute_accuracy(training_data, test_data, k, p)
        print(f"k={k}, p={p}: val_acc={acc_val:.3f}, test_acc={acc_test:.3f}")
        val_accuracies[i][j] = acc_val
        test_accuracies[i][j] = acc_test

# Plot heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Validation accuracy heatmap
sns.heatmap(val_accuracies, annot=True, fmt='.3f', 
            xticklabels=p_range, yticklabels=k_range,
            ax=ax1, cmap='viridis')
ax1.set_title('Validation Accuracy')
ax1.set_xlabel('p value')
ax1.set_ylabel('k value')

# Test accuracy heatmap
sns.heatmap(test_accuracies, annot=True, fmt='.3f',
            xticklabels=p_range, yticklabels=k_range,
            ax=ax2, cmap='viridis')
ax2.set_title('Test Accuracy')
ax2.set_xlabel('p value')
ax2.set_ylabel('k value')

plt.tight_layout()
plt.savefig('k_p_accuracy_heatmap.png')
plt.show()

# Accuracy before scale
# Compute accuracy
acc_val = compute_accuracy(training_data, validation_data, k, p)
acc_test = compute_accuracy(training_data, test_data, k, p)
print(f"Validation accuracy: {acc_val:.3f}")
print(f"Test accuracy: {acc_test:.3f}")

# %%
# Scale the data

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

# Add this after data loading in main():
training_data, validation_data, test_data = scale_data(training_data, validation_data, test_data)


