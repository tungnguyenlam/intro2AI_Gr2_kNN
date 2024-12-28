import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_distances(training_set, validation_point):
    distance_list = []
    for data in training_set:
        try:
            distance = sum((data[i] - validation_point[i])**2 for i in range(len(validation_point)))**0.5
            distance_list.append((distance, data[-1]))
        except IndexError:
            print(f"[ERROR] Mismatch in dimensions: Training point {data} vs Validation point {validation_point}")
            raise
    return distance_list


def classify_point(validation_point, distance_list, k):
    """
    Classify a validation point based on the k nearest points in distance_list.
    """
    if k > len(distance_list):
        raise ValueError(f"[ERROR] k ({k}) is larger than the training set size ({len(distance_list)}).")

    # Sort distances in ascending order
    distance_list.sort(key=lambda x: x[0])

    # Take k closest neighbors
    top_k = distance_list[:k]

    # Count each class frequency in top_k
    class_counts = {}
    for _, label in top_k:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Determine which class has the highest count
    predicted_class = max(class_counts, key=class_counts.get)
    return validation_point + [predicted_class]  # Return a copy with the predicted label


def apply_knn(k, training_set, validation_set):
    """
    Apply kNN to classify all points in validation_set.
    """
    predicted_points = []
    for idx, validation_point in enumerate(validation_set):
        print(f"\nProcessing validation point #{idx+1}/{len(validation_set)}: {validation_point}")

        # Calculate distances from validation_point to the entire training_set
        distance_list = calculate_distances(training_set, validation_point)
        print(f"  -> Calculated {len(distance_list)} distances.")

        # Classify validation_point based on the k nearest neighbors
        predicted_point = classify_point(validation_point, distance_list, k)
        print(f"  -> Predicted label for this point = {predicted_point[-1]}")

        # Store the classified point in the result list
        predicted_points.append(predicted_point)

    return predicted_points


def calculate_accuracy(validation_set, true_labels):
    """
    Compute the accuracy by comparing predictions with true labels.
    """
    correct_predictions = 0
    for i, validation_point in enumerate(validation_set):
        if validation_point[-1] == true_labels[i]:  # Compare predicted label with the true label
            correct_predictions += 1
    return correct_predictions / len(true_labels)


import csv

# Read CSV file
file_name = "country_wise_latest.csv"
data = []
try:
    with open(file_name, "r") as f:
        csv_f = csv.reader(f)
        header = next(csv_f)
        print("[INFO] CSV Header:", header)

        for row_idx, row in enumerate(csv_f):
            try:
                # Convert all but the last element to float and keep the last element as label
                features = list(map(float, row[:-1]))
                label = row[-1]
                data.append(features + [label])
            except ValueError as ve:
                print(f"[ERROR] Could not parse row #{row_idx}: {row}. Error: {ve}")
except FileNotFoundError:
    print(f"[ERROR] File not found: {file_name}")
    exit()

# Ensure we have enough data
if len(data) == 0:
    print("[ERROR] No valid data found in the file.")
    exit()

# Split into training_set and validation_set
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
training_set = data[:split_index]
validation_set = [row[:-1] for row in data[split_index:]]  # Validation set without labels
true_labels = [row[-1] for row in data[split_index:]]      # Save the actual labels for validation set

# Ensure split is valid
if len(training_set) == 0 or len(validation_set) == 0:
    print("[ERROR] Training or validation set is empty. Check your data split.")
    exit()

# k value
k = 2

# Apply kNN
print("\n[INFO] Starting kNN classification...")
try:
    predicted_validation_set = apply_knn(k, training_set, validation_set)
    accuracy = calculate_accuracy(predicted_validation_set, true_labels)
    print(f"\n[INFO] Finished kNN classification.")
    print(f"[RESULT] Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    print(f"[ERROR] An error occurred during kNN execution: {e}")


# Create the DataFrame from your dataset
df = pd.DataFrame(data, columns=header)

# Exclude non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[float, int])  # Keep only numeric columns

# Plot the heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.savefig('Correlation Matrix', dpi=300, bbox_inches='tight')
plt.show()
