import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

# Separate features and target
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define resampling techniques
resampling_techniques = {
    'No Resampling': None,
    'SMOTE': SMOTE(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42)
}

# Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Store all results for comparison
all_results = {}
best_combination = {'accuracy': 0, 'model': None, 'preprocessing': None, 'resampling': None}

# Define preprocessing techniques
preprocessing_techniques = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'LogTransformation': None  # Special handling for log transformation
}

# Train models with each combination of preprocessing and resampling
for preprocess_name, scaler in preprocessing_techniques.items():
    print(f"\n{'='*80}")
    print(f"PREPROCESSING: {preprocess_name}")
    print(f"{'='*80}")
    
    # Apply preprocessing
    if preprocess_name == 'LogTransformation':
        # Apply log transformation (add small constant to avoid log(0))
        X_train_processed = np.log1p(X_train)
        X_test_processed = np.log1p(X_test)
    else:
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    
    # Store results for each preprocessing technique
    all_results[preprocess_name] = {}
    
    # Apply each resampling technique
    for resample_name, resampler in resampling_techniques.items():
        print(f"\n  {resample_name}:")
        
        # Apply resampling to training data
        if resampler is None:
            X_train_resampled = X_train_processed
            y_train_resampled = y_train
        else:
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_processed, y_train)
        
        all_results[preprocess_name][resample_name] = {}
        
        # Train each model with this preprocessing and resampling
        for model_name, model in models.items():
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train_resampled, y_train_resampled)
            y_pred = model_clone.predict(X_test_processed)
            accuracy = accuracy_score(y_test, y_pred)
            
            all_results[preprocess_name][resample_name][model_name] = accuracy
            
            print(f"    {model_name}: {accuracy:.4f}")
            
            # Track best combination
            if accuracy > best_combination['accuracy']:
                best_combination['accuracy'] = accuracy
                best_combination['model'] = model_name
                best_combination['preprocessing'] = preprocess_name
                best_combination['resampling'] = resample_name
                best_combination['scaler'] = scaler if preprocess_name != 'LogTransformation' else None

# Find best model across all combinations
best_model_name = best_combination['model']
best_preprocessing = best_combination['preprocessing']
best_resampling = best_combination['resampling']
best_accuracy = best_combination['accuracy']

print(f"\n{'='*80}")
print(f"BEST COMBINATION")
print(f"{'='*80}")
print(f"Preprocessing: {best_preprocessing}")
print(f"Resampling: {best_resampling}")
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"{'='*80}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_combination['scaler'], f)

# Save results to JSON
with open('model_results.json', 'w') as f:
    json.dump({
        'best_combination': {
            'preprocessing': best_preprocessing,
            'resampling': best_resampling,
            'model': best_model_name,
            'accuracy': float(best_accuracy)
        },
        'all_results': all_results
    }, f, indent=2)

# Create comprehensive comparison table
print("\n" + "="*140)
print("COMPLETE MODEL, PREPROCESSING AND RESAMPLING COMPARISON")
print("="*140)

# Flatten results for display
comparison_data = []
for preprocess_name in all_results.keys():
    for resample_name in all_results[preprocess_name].keys():
        for model_name in models.keys():
            if model_name in all_results[preprocess_name][resample_name]:
                accuracy = all_results[preprocess_name][resample_name][model_name]
                comparison_data.append({
                    'Preprocessing': preprocess_name,
                    'Resampling': resample_name,
                    'Model': model_name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Accuracy %': f"{accuracy*100:.2f}%"
                })

comparison_df = pd.DataFrame(comparison_data)
comparison_df_sorted = comparison_df.sort_values(by='Accuracy', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))
print(comparison_df_sorted.to_string(index=False))
print("="*140)
