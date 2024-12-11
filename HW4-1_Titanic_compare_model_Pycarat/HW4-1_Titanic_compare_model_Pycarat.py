import pandas as pd
from pycaret.classification import setup, compare_models, stack_models, predict_model

# Load Titanic dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Drop rows with missing target values
train_data = data_train.dropna(subset=['Survived'])

# Define the setup for PyCaret
env = setup(
    data=train_data,
    target='Survived',
    categorical_features=['Pclass', 'Sex', 'Embarked'],
    numeric_features=['Age', 'Fare'],
    ignore_features=['PassengerId', 'Name', 'Ticket', 'Cabin']
)

# Compare and select top 16 models
best_models = compare_models(
    n_select=16,
    include=[
        'lr',      # Logistic Regression
        'knn',     # K Neighbors Classifier
        'nb',      # Naive Bayes
        'dt',      # Decision Tree Classifier
        'svm',     # SVM - Linear Kernel
        'rbfsvm',  # SVM - Radial Kernel
        'gpc',     # Gaussian Process Classifier
        'mlp',     # MLP Classifier
        'ridge',   # Ridge Classifier
        'rf',      # Random Forest Classifier
        'qda',     # Quadratic Discriminant Analysis
        'ada',     # Ada Boost Classifier
        'gbc',     # Gradient Boosting Classifier
        'lda',     # Linear Discriminant Analysis
        'et',      # Extra Trees Classifier
        'lightgbm',# Light Gradient Boosting Machine
    ]
)

# Perform Stacking using the top 16 models
stacked_model = stack_models(best_models)

# Predict on the test dataset
# Note: Test dataset needs similar preprocessing as training dataset. Adjust as necessary.
predictions = predict_model(stacked_model, data=data_test)

# Save the predictions
predictions[['PassengerId', 'prediction_label']].to_csv('titanic_predictions.csv', index=False)

print("Stacking complete. Predictions saved to 'titanic_predictions.csv'.")
