import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(filename, features, label_col):
    """
    Load and preprocess dataset from a CSV file.
    This function reads a CSV file, extracts the specified features and
    the label column, and returns them as NumPy arrays.
    """
    print("\n[load_data] Starting to load data from:", filename)
    print("[load_data] Features to extract:", features)
    print("[load_data] Label column:", label_col)
    
    data = []
    labels = []
    
    # Open the CSV file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        # Read the header row to identify the column indices
        header = next(reader)
        
        # Find the indices of the features and the label in the header
        feature_indices = [header.index(feat) for feat in features]
        label_index = header.index(label_col)
        
        print("[load_data] Identified feature indices:", feature_indices)
        print("[load_data] Identified label index:", label_index)
        
        # Iterate through each row in the CSV file
        for row in reader:
            try:
                # Extract numerical values for each feature
                feature_row = [float(row[i]) for i in feature_indices]
                # Extract label (as a string or category)
                label = row[label_index]
                # Append to the data and labels lists
                data.append(feature_row)
                labels.append(label)
            except ValueError:
                # If a row has invalid data (e.g., missing or non-numeric),
                # we skip that row
                continue  
    
    # Convert lists to NumPy arrays for easy scientific computing
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    print("[load_data] Finished loading data.")
    print("[load_data] Data shape:", data_array.shape)
    print("[load_data] Labels shape:", labels_array.shape)
    
    return data_array, labels_array

def compute_scatter_matrices(X, y, class_labels):
    """
    Compute within-class scatter matrix (Sw) and between-class scatter matrix (Sb).
    Sw measures how much the samples within each class vary, and
    Sb measures how far apart the different class means are.
    """
    print("\n[compute_scatter_matrices] Computing scatter matrices for LDA.")
    
    n_features = X.shape[1]
    # Overall mean of the entire dataset
    mean_overall = np.mean(X, axis=0)
    
    # Initialize Sw and Sb to zero matrices of shape (n_features, n_features)
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    
    # For each class, compute the class-specific statistics
    for label in class_labels:
        # Extract all samples belonging to the current label
        X_class = X[y == label]
        # Mean of the samples in the current class
        mean_class = np.mean(X_class, axis=0)
        
        # Center the data by subtracting the class mean
        X_centered = X_class - mean_class
        # Accumulate within-class scatter
        Sw += X_centered.T @ X_centered
        
        # Calculate how different this class mean is from the overall mean
        n_samples = X_class.shape[0]
        mean_diff = (mean_class - mean_overall).reshape(n_features, 1)
        # Accumulate between-class scatter
        Sb += n_samples * (mean_diff @ mean_diff.T)
    
    print("[compute_scatter_matrices] Sw shape:", Sw.shape)
    print("[compute_scatter_matrices] Sb shape:", Sb.shape)
    
    return Sw, Sb

def compute_discrimination_ratio(Sw, Sb, n_components):
    """
    Compute the discrimination ratio for the first n_components eigenvalues.
    The ratio is the sum of the top n_components eigenvalues of inv(Sw)*Sb
    divided by the sum of all eigenvalues. Higher ratio means more discriminatory power.
    """
    print(f"\n[compute_discrimination_ratio] Computing discrimination ratio for {n_components} components.")
    # Compute eigenvalues from the matrix pinv(Sw)*Sb
    eigvals, _ = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    # Only real parts matter
    eigvals = np.real(eigvals)
    # Sort eigenvalues in descending order
    eigvals = np.sort(eigvals)[::-1]
    
    # Sum of the top n_components eigenvalues divided by the sum of all
    ratio = np.sum(eigvals[:n_components]) / np.sum(eigvals)
    print(f"[compute_discrimination_ratio] Discrimination ratio = {ratio:.4f}")
    
    return ratio

def analyze_features(X, y, feature_names):
    """
    Analyze feature importance using LDA, by examining diagonal elements
    of Sw (within-class scatter) and Sb (between-class scatter).
    The ratio Sb[i, i] / Sw[i, i] indicates how discriminative feature i is.
    """
    print("\n[analyze_features] Analyzing feature importance using LDA.")
    # Identify unique class labels (already numeric in this case)
    class_labels = np.unique(y)
    # Compute scatter matrices
    Sw, Sb = compute_scatter_matrices(X, y, class_labels)
    
    # Calculate a discrimination score for each feature
    feature_scores = []
    for i in range(X.shape[1]):
        sw_i = Sw[i, i]
        sb_i = Sb[i, i]
        ratio = sb_i / sw_i if sw_i != 0 else 0
        feature_scores.append((feature_names[i], ratio))
    
    # Sort features by their discrimination ratio
    feature_scores_sorted = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    
    print("[analyze_features] Feature scores (unsorted):", feature_scores)
    print("[analyze_features] Feature scores (sorted):", feature_scores_sorted)
    
    return feature_scores_sorted

def main():
    print("\n=== START OF MAIN FUNCTION ===")
    
    # Define file and attributes
    print("[main] Defining dataset file and the list of features and label to extract.")
    dataset_file = 'final_pollution_dataset.csv'
    features = [
        'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 
        'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'
    ]
    label_col = 'Air Quality'
    
    print("[main] Loading data from CSV using load_data function.")
    # Load data
    X, y = load_data(dataset_file, features, label_col)
    
    print("\n[main] Standardizing the features to have mean=0 and std=1.")
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[main] Encoding class labels as integers for easier processing.")
    # Encode class labels as integers
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y])
    
    print("[main] Computing scatter matrices for the scaled data.")
    # Compute scatter matrices
    class_labels = np.unique(y_int)
    Sw, Sb = compute_scatter_matrices(X_scaled, y_int, class_labels)
    
    print("\n[main] Evaluating discrimination ratios for each possible dimension (1 to max_dims).")
    # Compute discrimination ratios for dimensions
    max_dims = X.shape[1]
    ratios = []
    for dim in range(1, max_dims + 1):
        ratio = compute_discrimination_ratio(Sw, Sb, dim)
        ratios.append(ratio)
        print(f"[main] Discrimination ratio for {dim}D = {ratio:.4f}")
    
    print("\n[main] Analyzing feature importance.")
    # Analyze feature importance
    feature_scores = analyze_features(X_scaled, y_int, features)
    
    # Print feature analysis
    print("\nFeature Importance Ranking:")
    print("---------------------------")
    for rank, (feature, score) in enumerate(feature_scores, 1):
        print(f"{rank}. {feature}: {score:.4f}")
    
    print("\n[main] Plotting discrimination ratios vs. number of dimensions.")
    # Plot discrimination ratios
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_dims + 1), ratios, 'bo-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Discrimination Ratio')
    plt.title('LDA Discrimination Ratio vs. Dimensions')
    plt.grid(True)
    plt.savefig('lda_analysis.png')
    print("[main] Discrimination ratio plot saved as lda_analysis.png.")
    plt.show()
    
    print("\n[main] Based on feature importance scores, recommending top features for kNN.")
    # Recommend features for kNN
    top_2 = [score[0] for score in feature_scores[:2]]
    top_3 = [score[0] for score in feature_scores[:3]]
    print("Top 2 features:", top_2)
    print("Top 3 features:", top_3)
    
    print("\n=== END OF MAIN FUNCTION ===")

if __name__ == "__main__":
    main()