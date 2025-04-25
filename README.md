# deep-learning-challenge




**Predicting Success in Charity Donations**

Purpose of the Analysis

The goal of this project is to build a binary classification model that predicts whether applicants for funding from charitable organizations will be successful. This prediction is based on features such as application type, classification, organization, and various categorical and numeric features from the `charity_data.csv` dataset.

By preprocessing the data, engineering features, and training a neural network model, we aim to provide a decision-making tool that can help non-profits allocate resources more effectively.

---
#Results#

**1. How many features were used to train the model?**

After preprocessing (dropping unnecessary columns, encoding categorical variables, and removing low-frequency values), the dataset was converted using one-hot encoding. The final input shape to the model was:
```python
X_train_scaled.shape[1]
```
This value corresponds to **116 features**, which were used to train the deep learning model.

**2. How many hidden layers were in the model?**

The model architecture included:
- 1st hidden layer with **80 nodes**, using **ReLU** activation
- 2nd hidden layer with **30 nodes**, also using **ReLU**
- Output layer with **1 node** and **sigmoid** activation for binary classification

**3. What was the activation function used on the output layer?**

The output layer used the **sigmoid** activation function, which is appropriate for binary classification tasks like predicting success (0 or 1).

**4. What was the accuracy achieved by the model?**

After training for 20 epochs, the model achieved the following on the test data:
```python
Loss: 0.5442, Accuracy: 0.7277
```
This corresponds to an accuracy of **~72.8%**.

**5. What loss function was used?**

The model used **binary crossentropy** (`loss="binary_crossentropy"`), which is the standard loss function for binary classification tasks.

**6. Was the model exported?**

Yes, the final model was exported using:
```python
nn.save("charity_model.h5")
```
This HDF5 file contains the trained model's architecture, weights, and configuration for reuse.

##Summary of the Model Results

The neural network achieved nearly **73% accuracy** on unseen test data, which indicates good generalization given the dataset size and structure. It is a solid baseline for identifying successful applications, particularly after feature engineering steps like grouping rare categories and scaling input data.

However, performance could likely be improved by further hyperparameter tuning (e.g., number of nodes, layers, epochs), and advanced techniques like dropout or batch normalization.

##Alternative Model Consideration

To potentially improve performance and reduce training time, an alternative model such as a **Random Forest Classifier** could be used. Random forests handle categorical features well (especially after encoding), are resistant to overfitting, and provide feature importance metrics, making them highly interpretable.

Benefits of using a random forest:
- Handles non-linear data better with minimal tuning
- Offers built-in feature importance scoring
- Can outperform shallow neural nets on tabular data

Thanks for reading!
