import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(filename, features, label_col):
    """Load and preprocess dataset from a CSV file."""
    data = []
    labels = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        feature_indices = [header.index(feat) for feat in features]
        label_index = header.index(label_col)
        
        for row in reader:
            try:
                feature_row = [float(row[i]) for i in feature_indices]
                label = row[label_index]
                data.append(feature_row)
                labels.append(label)
            except ValueError:
                continue  # Skip rows with invalid data
    
    return np.array(data), np.array(labels)

def compute_scatter_matrices(X, y, class_labels):
    """Compute within-class and between-class scatter matrices."""
    n_features = X.shape[1]
    mean_overall = np.mean(X, axis=0)
    
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    
    for label in class_labels:
        X_class = X[y == label]
        mean_class = np.mean(X_class, axis=0)
        
        X_centered = X_class - mean_class
        Sw += X_centered.T @ X_centered
        
        n_samples = X_class.shape[0]
        mean_diff = (mean_class - mean_overall).reshape(n_features, 1)
        Sb += n_samples * (mean_diff @ mean_diff.T)
    
    return Sw, Sb

def compute_discrimination_ratio(Sw, Sb, n_components):
    """Compute discrimination ratio for a given number of components."""
    eigvals, _ = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    eigvals = np.real(eigvals)
    eigvals = np.sort(eigvals)[::-1]  # Sort in descending order
    
    return np.sum(eigvals[:n_components]) / np.sum(eigvals)

def analyze_features(X, y, feature_names):
    """Analyze feature importance using LDA."""
    class_labels = np.unique(y)
    Sw, Sb = compute_scatter_matrices(X, y, class_labels)
    
    # Compute per-feature discrimination
    feature_scores = []
    for i in range(X.shape[1]):
        sw_i = Sw[i, i]
        sb_i = Sb[i, i]
        ratio = sb_i / sw_i if sw_i != 0 else 0
        feature_scores.append((feature_names[i], ratio))
    
    return sorted(feature_scores, key=lambda x: x[1], reverse=True)

def main():
    # Define file and attributes
    dataset_file = 'final_pollution_dataset.csv'
    features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 
                'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
    label_col = 'Air Quality'
    
    # Load data
    X, y = load_data(dataset_file, features, label_col)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode class labels as integers
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y])
    
    # Compute scatter matrices
    class_labels = np.unique(y_int)
    Sw, Sb = compute_scatter_matrices(X_scaled, y_int, class_labels)
    
    # Compute discrimination ratios for dimensions
    max_dims = X.shape[1]
    ratios = []
    for dim in range(1, max_dims + 1):
        ratio = compute_discrimination_ratio(Sw, Sb, dim)
        ratios.append(ratio)
        print(f"Discrimination ratio for {dim}D: {ratio:.4f}")
    
    # Analyze feature importance
    feature_scores = analyze_features(X_scaled, y_int, features)
    
    # Print feature analysis
    print("\nFeature Importance Ranking:")
    print("---------------------------")
    for rank, (feature, score) in enumerate(feature_scores, 1):
        print(f"{rank}. {feature}: {score:.4f}")
    
    # Plot discrimination ratios
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_dims + 1), ratios, 'bo-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Discrimination Ratio')
    plt.title('LDA Discrimination Ratio vs Dimensions')
    plt.grid(True)
    plt.savefig('lda_analysis.png')
    plt.show()
    
    # Recommend features for kNN
    print("\nRecommended features for kNN:")
    print("Top 2 features:", [score[0] for score in feature_scores[:2]])
    print("Top 3 features:", [score[0] for score in feature_scores[:3]])

if __name__ == "__main__":
    main()
