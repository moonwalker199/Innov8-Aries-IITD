# Part 2: Predicting Troop Betrayal in the War Against the Phrygians

## Submitted by **team grey**

      1. Akshat Namdeo
      2. Debangan Sarkar
      3. Abhinav Singh Naruka
      4. Ashi Gupta

      Indian Institute of Technology Roorkee.

## Project Overview

This project aims to predict which soldiers in your army are most likely to betray the clan using statistical modeling and machine learning techniques. Features such as **greed**, **temptation**, **loyalty**, **respect**, and **peer influence** are hypothesized to influence the likelihood of betrayal.

## **Key Factor Hypothesis**:

To predict a soldier's potential for betrayal, we can hypothesize based on various influencing factors as mentioned below:

- **Greed**: Desire for wealth or power
- **Temptation**: External rewards may entice
- **Poor respect**: Feeling undervalued leads to disloyalty
- **Loyalty history**: If a soldier has shown signs of loyalty before.
- **Economic hardship**: Financial strain increases temptation
- **Social ties**: Any connection with the traitors(Phygrians).
- **Distance from command**: Farther from leadership, weaker loyalty.
- **Performance rating**: Low performers might have less loyalty.
- **Mission failures**: A history of failed missions might indicate lower commitment.
- **Disciplinary actions**: Soldiers with disciplinary issues may be more likely to defect.<br>

etc.

## **Quantification and Feature Selection**:

Based on the collected data about the above mentioned features, we will be quantifying them using **statistical modelling**. <br>

We integrate hypothesis testing to evaluate the significance of features related to betrayal risk and then train our model using those features.

```python
results = {}
for feature in df.columns[:-1]:  # Exclude the target variable
    loyal_scores = df[df['betrayal_risk'] == 0][feature]
    betrayal_scores = df[df['betrayal_risk'] == 1][feature]

    # Perform Z-test
    z_stat, p_value = stats.ttest_ind(loyal_scores, betrayal_scores, equal_var=False)
    results[feature] = {
        'Z-statistic': z_stat,
        'P-value': p_value,
        'Significant': p_value < 0.05
    }
#significant feature selection
significant_features = [feature for feature, result in results.items() if result['Significant']]
X_significant = df[significant_features]
```

# **WORKFLOW OF OUR SYSTEM**

## Step 1: Data Preparation

- _Dataset Creation_: A dataset is created containing these features:

  - greed: A numerical score indicating the level of greed.
  - loyalty: A numerical score reflecting loyalty.
  - temptation: A numerical score representing temptation levels.
  - performance: A numerical score depicting performance.
  - betrayal_risk: The target variable, where 0 indicates loyalty and 1 indicates betrayal risk.

- _DataFrame_: The dataset is structured as a Pandas DataFrame, allowing for easy manipulation and analysis.

## Step 2: Statistical Analysis

- _Z-Test_: For each feature (excluding the target variable), a Z-test is performed to compare the means of the loyal and betrayal risk groups.

  - _Hypothesis_:
    - Null Hypothesis (H0): There is no significant difference in the means of the two groups.
    - Alternative Hypothesis (H1): There is a significant difference in the means of the two groups.
  - _Results_: The Z-statistic and p-value (alpha) are calculated. If alpha < 0.05, the feature is considered statistically significant.

- _Results Summary_: The Z-statistics and significance levels are printed for each feature, providing insight into which features may influence betrayal risk.

## Step 3: Feature Selection

- _Significant Features_: Features that have shown statistical significance in the previous step are selected for further analysis. This reduces dimensionality and focuses on the most impactful variables.

## Step 4: Data Preprocessing

- _Standardization_: The selected features are standardized using StandardScaler. This transforms the data to have a mean of 0 and a standard deviation of 1, ensuring that all features contribute equally to the model training.

## Step 5: Handling Class Imbalance

- _SMOTE_: To address potential class imbalance (where one class may have significantly fewer samples), the Synthetic Minority Over-sampling Technique (SMOTE) is applied. This generates synthetic samples for the minority class (betrayal risk) to balance the dataset, improving model performance.

## Step 6: Model Training and Hyperparameter Tuning

- _Train-Test Split_: The dataset is divided into training and testing sets using an 80-20 split, ensuring that the model can be evaluated on unseen data.

- _Model Selection_: A RandomForestClassifier is chosen as the base model due to its robustness and effectiveness in classification tasks.

- _Hyperparameter Tuning_: GridSearchCV is utilized to find the best hyperparameters for the model. The parameters being tuned include:

  - n_estimators: Number of trees in the forest.
  - max_depth: Maximum depth of each tree.
  - min_samples_split: Minimum number of samples required to split an internal node.

- _Training_: The model is trained using the training data, and the best hyperparameters are determined through cross-validation.

## Step 7: Model Evaluation

- _Prediction_: The best model is used to predict outcomes on the test set.

- _Performance Metrics_: The model's performance is assessed using two key metrics:

  - _Accuracy_: The proportion of correctly predicted instances out of the total instances.
  - _F1 Score_: The harmonic mean of precision and recall, providing a balance between the two, especially useful for imbalanced datasets.

- _Results Output_: The optimized model's accuracy and F1 score are printed, giving an indication of how well the model can predict betrayal risk based on the input features.

## Step 8: Scalability

To enhance scalability, the system should implement continuous data collection and regularly retrain the model in response to emerging factors, such as the introduction of new soldiers and evolving war conditions. Automating this process will allow the system to adapt and improve over time.

## **EXPLANATION OF FULL-STACK**

### Languages

- **Python**: The entire pipeline is written in Python, a widely used language for data analysis and machine learning.

### Python Libraries Used

- **Pandas** (`pd`): For data manipulation and reading the CSV dataset.
- **NumPy** (`np`): For efficient numerical operations.
- **SciPy** (`scipy.stats`): For performing statistical analysis (t-tests).
- **Scikit-learn** (`sklearn`): For machine learning models, data preprocessing, and model evaluation.
- **Imbalanced-learn** (`imblearn`): To handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).



### Additional Considerations

### **Model Explainability**

- Use tools like **SHAP** (SHapley Additive exPlanations) to interpret the model's predictions and understand the contribution of each feature.

### **Model Monitoring**

- Once deployed, monitor the model performance to detect data drift or performance degradation over time, ensuring the model continues to perform well.

---

## Running the Code

1. **Install dependencies**:

   ```terminal
   pip install pandas numpy scipy scikit-learn imbalanced-learn
   ```

2. **Execute the code**:

   - Ensure the CSV file `soldier.csv` is in the same directory as the script.
   - Run the Python script.

3. **Hyperparameter Optimization**:

   - The hyperparameter tuning process will take time depending on the dataset size. The optimized model will be evaluated and metrics displayed.

4. **Modify for your dataset**:
   - Adjust the code to handle different datasets by modifying the data loading and preprocessing sections.

---

## Conclusion

This project demonstrates a robust machine learning pipeline for binary classification, handling statistical analysis, feature selection, class imbalance, and hyperparameter tuning. The stack is designed for flexibility, scalability, and ease of deployment.

## **SAMPLE CODE**

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats

df = pd.read_csv('soldier.csv')

#Statistical Analysis using Z-Test
results = {}
for feature in df.columns[:-1]:  # Exclude the target variable
    loyal_scores = df[df['betrayal_risk'] == 0][feature]
    betrayal_scores = df[df['betrayal_risk'] == 1][feature]

       z_stat, alpha = stats.ttest_ind(loyal_scores, betrayal_scores, equal_var=False)
    results[feature] = {
        'Z-statistic': z_stat,
        'alpha': alpha,
        'Significant': alpha < 0.05
    }

# Display statistical results
print("Statistical Analysis Results:")
for feature, result in results.items():
    print(f'{feature} - Z-statistic: {result["Z-statistic"]:.4f}, alpha: {result["alpha"]:.4f}, Significant: {result["Significant"]}')

#Select Significant Features
significant_features = [feature for feature, result in results.items() if result['Significant']]
X_significant = df[significant_features]
y = df['betrayal_risk']

#Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_significant)

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Hyperparameter Tuning with Randomized Search
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Applying the ML model
random_search = GridSearchCV(RandomForestClassifier(random_state=42), param_dist, cv=5)
random_search.fit(X_train, y_train)

# best model evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# printing the outputs
print(f'\nOptimized Model Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1:.2f}')
```
