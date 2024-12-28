# intro2AI_Gr2_kNN
This is the repository for the Final Group Project (Introduction to AI) on k-nearest neighbors (Group 2)
Members: (Everyone put your info here: Github username - Full name - StudentID - USTH email address)
hngoc2204 - Nguyen Vu Hong Ngoc - 23BI14345 - ngocnvh.23bi14345@usth.edu.vn
usthTonyNguyen - Nguyen Lam Tung - 23BI14446 - tungnl.23bi14446@usth.edu.vn
Palm-Pham - Pham Dinh Bao Khoi - 23BI 
watermelon-3012 - Le Sy Han - 23BI14150 - hanls.23bi14150@usth.edu.vn

Project requirements: (Need to improve)
- Implement k-nearest neighbors on a new dataset and train a model on it
- Dataset: Include 3 part
	 1. Training set: This is the dataset that the model is trained on 
	 2. Validation set: The dataset to evaluate the model performance before going to the real test, normally, if the model is failed in the validation, it should be reevaluated and retrained.
	 3. Testing set: What is the expected output? -> The model accuracy! If the accuracy is too low, then make improvment on the algorithm (aka code) and the data (quality? quantity?). Normally, the acc should not be too high (Surely < 100%, but from the lecturer ... Viet, it should falls btw 30% - 90%).

## Guide to use the code
The output and conclusion of Methodology in the Final Report follow the following code including 6 steps

# Step 1: Preprocess the Dataset
- "updated_pollution_dataset.csv" This file contains the original dataset
- The file "preprocess.ipynb" consists of 4 main functions: Shuffle the dataset (Run cell 1), Clean negative values (Run cell 2). This 2 cell is what produce the file "final_pollution_dataset.csv", which we use. Scale the data (Run cell 3), Add noise - Data Augmentation (Run cell 4). This two final cell is just for demonstration, only run when you know what you are doing. This file only need to run once since we need CONSISTENT dataset. 
- The file "final_pollution_dataset.csv" -> The output after preprocessing, this file and other files with the same name in other steps are identical. (This is why preprocess.ipynb run only once)
- "Splitdata.ipynb" - This file split the dataset into 3 file "Test_set.csv", "Training_set.csv", "Validating_set.csv" -> This is just for demonstrative purposes, in actual case, the whole dataset is converted into 2 dimenstional array and the training work only on that file (By copying part of an array to another array, we can create training_set, validation_set and test_set)

# Step 2: Correlation Matrix (3 file)
- "correlation_matrix.py" - This file read the data in "final_pollution_dataset.csv" and create an Correlation Matrix that show how independent the variables (attributes) in the dataset is.
- "Correlation Matrix.png" This file is the output of "correlation_matrix.py", this figure is then used in the report in Methodology.

# Step 3: Linear Discrimination Analysis
- "lda2.py" - Read the dataset file and produce 2 output
	+ Feature Importance Ranking Score (terminal): This is turn into a table in the report (Methodology)
	+ "lda_analysis.png" - The image file that plot the LDA Discriminatoin vs Dimension -> Basicly tell us the lowest dimension we can reduce the dataset into
- "manual_lda.py": Same function, but print the plot data in number
- More: 
	+ Step 2 and Step 3 combine provide us a hint on how to choose the right number and combination of attribute that we should work with
	+ Normaly, the number 3 - dimension that we mention does not directly corrolate with 3 attributes, but 3 dimension that hard-coded the information of all 9 attribute. But for the sake of simplicity, we decided to implt 3 dimension directly

# Step 4: Plot the dataset
- Straight forward
- "plotdata3d.py" - Read the dataset and plot on chosen attribute (If you want to plot another set, just modify the corresponding name) -> Output "plot.png"
- plot_covid: Directory to plot the covid dataset
	+ Improved code, easier to modify, just modify one list
- plot_iris: Same as plot_covid
- Guide to plot: Choose attribute to plot in the dataset by using Correlation Matrix -> Choose combination that their correlation not too high. -> Modify the code (file name, choosen attributes)

# Step 5: Plot accuracy of model with k form 1->20
- "___.csv" -> Dataset used
- "plot_k.py": Read the training data, validation data, test data -> Calculate the validation accuracy, test accuracy of k from 1->20 and plot it 
- Can change the ploting range by changing the range of k

# Step 6: Plot the classification map
- "classification_map.py" -> When run, plot the decision map of the dataset with chosen attributes.
- if we choose 2 attribute, plot decision map for those 2 att
- if choose 3 attributes, plot decision map in 3 angles (1-2, 2-3, 1-3)
- decision map for 2 attribute will help to classify a given point
- decision map for 3 attribute WILL NOT help to classify a point, because we are essentially taking 3 slices of the decision space
