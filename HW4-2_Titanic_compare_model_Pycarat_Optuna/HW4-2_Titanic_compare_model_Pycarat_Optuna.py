import pandas as pd
from pycaret.classification import setup, compare_models, stack_models, predict_model, tune_model, create_model

# Load Titanic dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Feature engineering
def simplify_ages(df):
    df['Age'] = df['Age'].fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    df['Age'] = pd.cut(df['Age'], bins, labels=group_names)
    return df

def simplify_fares(df):
    df['Fare'] = df['Fare'].fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    df['Fare'] = pd.cut(df['Fare'], bins, labels=group_names)
    return df

def simplify_cabins(df):
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    return df

def format_name(df):
    df['Lname'] = df['Name'].apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df['Name'].apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_fares(df)
    df = simplify_cabins(df)
    df = format_name(df)
    df = drop_features(df)
    return df

train_data = transform_features(data_train)
data_test = transform_features(data_test)

# Drop rows with missing target values
train_data = train_data.dropna(subset=['Survived'])

# Define the setup for PyCaret
env = setup(
    data=train_data,
    target='Survived',
    categorical_features=['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Lname', 'NamePrefix'],
    numeric_features=[],
    ignore_features=['PassengerId'],
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

# Optimize top models using Optuna
tuned_models = [tune_model(model, search_library='optuna', search_algorithm='random') for model in best_models]

# Perform Stacking using the optimized models
stacked_model = stack_models(tuned_models)

# Predict on the test dataset
predictions = predict_model(stacked_model, data=data_test)

# Save the predictions
predictions[['PassengerId', 'prediction_label']].to_csv('titanic_predictions_optimized.csv', index=False)

print("Optimization and stacking complete. Predictions saved to 'titanic_predictions_optimized.csv'.")
