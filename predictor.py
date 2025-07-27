import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class FireRiskPredictor:
    def __init__(self, data_path):
        
        # Load the dataset
        self.data = pd.read_csv(data_path)
        
        # Print column names to check for discrepancies
        print("Column names in the dataset:")
        print(self.data.columns)
        
        # Convert the 'area' column to binary (0 = no fire, 1 = fire)
        self.data['fire'] = (self.data['area'] > 0).astype(int)
        
        # Update target to the binary 'fire' column
        self.features = [
            'temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI'
        ]
        self.target = 'fire'  # Updated target column
        
        # Preprocessing flags
        self.is_preprocessed = False
        self.is_trained = False
        
        # Model and scaler placeholders
        self.model = None
        self.scaler = None
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        
        # Check for missing values
        print("Missing values before preprocessing:")
        print(self.data[self.features + [self.target]].isnull().sum())
        
        # Handle missing values (simple strategy: drop rows with missing values)
        self.data.dropna(subset=self.features + [self.target], inplace=True)
        
        print("\nMissing values after preprocessing:")
        print(self.data[self.features + [self.target]].isnull().sum())
        
        # Separate features and target
        X = self.data[self.features]
        y = self.data[self.target]
        
        # Check the class distribution for 'fire' (target)
        print("\nClass distribution before split:")
        print(y.value_counts())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        self.is_preprocessed = True
        print("\nData preprocessing completed successfully.")
    
    def train_model(self, n_estimators=100, random_state=42):
        
        if not self.is_preprocessed:
            raise ValueError("Please preprocess the data first using preprocess_data() method")
        
        # Initialize and train the Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            class_weight='balanced'  # Handles class imbalance
        )
        self.model.fit(self.X_train, self.y_train)
        
        self.is_trained = True
        print("Model training completed successfully.")
    
    def evaluate_model(self):
        
        if not self.is_trained:
            raise ValueError("Please train the model first using train_model() method")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Compute metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1 Score': f1_score(self.y_test, y_pred),
            'ROC AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Print metrics
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fire', 'Fire'], 
                    yticklabels=['No Fire', 'Fire'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def predict_fire_risk(self, new_data, threshold=0.5):
        
        if not self.is_trained:
            raise ValueError("Please train the model first using train_model() method")
    
        # Convert input to DataFrame if it's a dictionary
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in new_data.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select and scale the features
        X_new = new_data[self.features]
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict
        prediction_proba = self.model.predict_proba(X_new_scaled)[:, 1]
        
        # Determine risk level based on the threshold
        risk_level = "High Risk" if prediction_proba[0] > threshold else "Low Risk"
        
        return {
            'prediction': int(prediction_proba[0] > threshold),
            'risk_level': risk_level,
            'fire_probability': float(prediction_proba[0])
        }
