import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
import matplotlib.pyplot as plt

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
    """
    Multiprocessing helper function.
    args: (i, j, k, p, training_data, validation_data, test_data)
    Returns: (i, j, val_acc, test_acc)
    """
    i, j, k, p, training_data, validation_data, test_data = args
    val_acc = compute_accuracy(training_data, validation_data, k, p)
    test_acc = compute_accuracy(training_data, test_data, k, p)
    return i, j, val_acc, test_acc

# -------------------------------------------------------------------
# 2. Main guard to prevent issues on Windows/macos spawn methods.
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Specify your CSV file and attributes here
    dataset_file = 'final_pollution_dataset.csv'
    attributes = ['SO2', 'CO', 'Proximity_to_Industrial_Areas']  

    # Split ratios
    training_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.2

    # Load the data
    chosen_header, training_data, validation_data, test_data, labels = initData(
        filename=dataset_file,
        training_ratio=training_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        attributes=attributes
    )

    # Define your k and p ranges
    k_values = list(range(1, 20))  # k from 1 to 9
    p_values = list(range(1, 10))   # p from 1 to 5

    # Create matrices to store validation & test accuracies
    val_accuracies = np.zeros((len(k_values), len(p_values)))
    test_accuracies = np.zeros((len(k_values), len(p_values)))

    # -------------------------------------------------------------------
    # 3. Prepare the argument list for parallel computations.
    # -------------------------------------------------------------------
    args_list = []
    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            args_list.append((i, j, k, p, training_data, validation_data, test_data))

    # -------------------------------------------------------------------
    # 4. Run parallel computations using ProcessPoolExecutor
    # -------------------------------------------------------------------
    total_tasks = len(args_list)
    print(f"Starting parallel computation for {total_tasks} (k, p) combinations...")

    with ProcessPoolExecutor() as executor:
        # We iterate as results come in
        for idx, (i, j, val_acc, test_acc) in enumerate(executor.map(compute_accuracies_for_k_p, args_list)):
            val_accuracies[i][j] = val_acc
            test_accuracies[i][j] = test_acc

            # Optional: print progress every 5 tasks
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{total_tasks} tasks...")

    # -------------------------------------------------------------------
    # 5. Print final results
    # -------------------------------------------------------------------
    print("\nFinal Results:")
    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            print(f"k={k}, p={p} -> val_acc={val_accuracies[i][j]:.3f}, test_acc={test_accuracies[i][j]:.3f}")

    print("\nValidation Accuracies Matrix:")
    print(val_accuracies)

    print("\nTest Accuracies Matrix:")
    print(test_accuracies)

# Plot heatmaps
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
plt.savefig('k_p_accuracy_heatmap.png')
plt.show()

# Accuracy before scale
# Compute accuracy
acc_val = compute_accuracy(training_data, validation_data, k, p)
acc_test = compute_accuracy(training_data, test_data, k, p)
print(f"Validation accuracy: {acc_val:.3f}")
print(f"Test accuracy: {acc_test:.3f}")