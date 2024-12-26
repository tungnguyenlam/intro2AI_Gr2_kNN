import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

def minkowski_distance(point_1, point_2, p):
    return sum(abs(a - b)**p for a, b in zip(point_1, point_2))**(1/p)

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

def compute_accuracy(training_data, data, k, p):

    if not data:
        return 0
    correct = sum(
        knn(training_data, point[:-1], k, p) == point[-1]
        for point in data
    )
    return correct / len(data)

def plot_decision_map(training_data, attributes, k, p, grid_step):
    print("[DEBUG] Generating decision map... This can take time if the range is large.")
    
    # Extract x, y, and labels from the training data
    x_vals = [row[0] for row in training_data]
    y_vals = [row[1] for row in training_data]
    data_labels = [row[-1] for row in training_data]
    
    # Define the plot range
    x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
    y_min, y_max = min(y_vals) - 1, max(y_vals) + 1
    
    # Generate a grid of points for the decision map
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    
    # Classify each grid point
    Z = np.array([
        knn(training_data, [mx, my], k, p)
        for mx, my in zip(xx.ravel(), yy.ravel())
    ]).reshape(xx.shape)
    
    # Map labels to integers for coloring
    unique_labels = sorted(set(data_labels))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    color_map = ListedColormap(colors)
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    Z_int = np.vectorize(label_to_idx.get)(Z)
    
    # Plot decision regions
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.pcolormesh(
        xx, yy, Z_int,
        cmap=color_map, alpha=0.2, shading='auto'
    )
    
    # Plot training points
    for label in unique_labels:
        mask = [row[-1] == label for row in training_data]
        x = [row[0] for row, m in zip(training_data, mask) if m][:100]
        y = [row[1] for row, m in zip(training_data, mask) if m][:100]
        ax.scatter(
            x, y,
            c=[color_map(label_to_idx[label])],
            label=label,
            edgecolor='black', linewidth=0.1,
            s=10
        )
    
    # Labels, title, and legend
    ax.set_xlabel(attributes[0], fontsize=12)
    ax.set_ylabel(attributes[1], fontsize=12)
    ax.set_title(f"Decision Boundaries: {attributes[0]} vs {attributes[1]}\nk={k}, p={p}", fontsize=14, pad=15)
    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save and display the figure
    plt.tight_layout()
    out_filename = f"decision_map_{attributes[0]}_{attributes[1]}_k{k}_p{p}.png"
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"[DEBUG] Decision map saved as '{out_filename}'")
    
    plt.show()
    plt.close()

def main():
    dataset_file = 'updated_pollution_dataset.csv'
    attributes = ['Temperature', 'SO2']  # Two selected attributes
    k = 6
    p = 4
    grid_step = 1.0  # Grid step size for faster computation
    
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
    
    # Plot decision map
    plot_decision_map(training_data, attributes, k, p, grid_step)

if __name__ == '__main__':
    main()
