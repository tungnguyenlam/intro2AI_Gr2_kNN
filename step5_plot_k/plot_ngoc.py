import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Load training data
training = pd.read_csv("FinalTrain_set.csv", usecols=['SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Air Quality'])
training_array = training[['SO2', 'CO', 'Proximity_to_Industrial_Areas']].to_numpy()
category_array = training['Air Quality'].to_numpy()


#Load validating data
validating= pd.read_csv("FinalVal_set.csv", usecols= ['SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Air Quality'])
validating_array= validating[['SO2', 'CO', 'Proximity_to_Industrial_Areas']].to_numpy()
valcat_array= validating['Air Quality'].to_numpy()

#Load testing data
testing= pd.read_csv("FinalTest_set.csv", usecols= ['SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Air Quality'])
test_array= testing[['SO2', 'CO', 'Proximity_to_Industrial_Areas']].to_numpy()
testcat_array= testing['Air Quality'].to_numpy()


# Minkowski distance function (p=3)
def minkowski_distance(x, y):
    p=3
    sum=0
    for i in range(len(x)):
        sum += abs(x[i] - y[i])**p
    return sum ** (1 / p)

# K-Nearest Neighbors (KNN) function
def KNN(trainSet, trainLabels, point, k):
    distances = []

    # Calculate distances between the point and all training data
    for i in range(len(trainSet)):
        dist = minkowski_distance(trainSet[i], point)
        #add distances to the list
        distances.append((dist, trainLabels[i]))

    # Sort distances in ascending order and select top k
    distances.sort()
    top_k_distances = distances[:k]

    # Count occurrences of each category in the top k
    counts = {'Good': 0, 'Moderate': 0, 'Poor': 0, 'Hazardous': 0}
    # donot need distance value in the loop so use _ to ignore
    for _, category in top_k_distances:
        counts[category] += 1

    # Determine the category with the highest count

    #use key= counts.get to retrieve the value for a given key
    return max(counts, key=counts.get)

# Predict function for validation or testing set based on training set
def predict(valSet, trainSet, trainLabels, k):
    val_pred = []
    for i in range(len(valSet)):
        pred = KNN(trainSet, trainLabels, valSet[i], k)
        val_pred.append(pred)
    return val_pred
    
# Scale the features for better performance
scaler= StandardScaler()
s_train_scaled= scaler.fit_transform(training_array)
s_test_scaled= scaler.fit_transform(test_array)
s_val_scaled= scaler.fit_transform(validating_array)
# Perform prediction on the validation and test sets
ks = np.arange(1,31)
accuracy_train={}
accuracy_val={}
accuracy_test={}
for k in ks:
    
    val_pred = predict(s_val_scaled, s_train_scaled, category_array, k)
    test_pred = predict(s_test_scaled, s_train_scaled, category_array, k)
    print("k=",k)
    
    # Calculate accuracy of validation set
    accuracy_val[k] = accuracy_score(valcat_array, val_pred)
    print("Validation accuracy: ", accuracy_val[k])
    # Calculate accuracy of test set
    accuracy_test[k] = accuracy_score(testcat_array, test_pred)
    print("Testing accuracy: ", accuracy_test[k])

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot validation accuracies
plt.plot(ks, accuracy_val.values(), label="Validation Accuracy")

# Plot test accuracies
plt.plot(ks, accuracy_test.values(), label="Testing Accuracy")

# Set integer ticks on x-axis
plt.xticks(ks)  # This will show only the k values as integers
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend()
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")

# Display the plot
plt.show()