import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

class MentalStateClassifier:
    """
    Comprehensive ML Model for EEG Mental State Classification
    Classifies REST vs MATH tasks using extracted features
    """
    
    def __init__(self):
        self.features_df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = None
        
    def load_features(self, features_file='eeg_features.csv'):
        """Load the extracted features"""
        try:
            self.features_df = pd.read_csv(features_file)
            print(f" Features loaded: {self.features_df.shape}")
            print(f"Task distribution:\n{self.features_df['task_type'].value_counts()}")
            return True
        except FileNotFoundError:
            print(f" Features file '{features_file}' not found")
            return False
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        if self.features_df is None:
            print(" No features loaded")
            return False
        
        # Remove non-feature columns
        exclude_columns = ['file_name', 'task_type', 'subject_id', 'duration', 'num_channels', 'sampling_rate']
        feature_columns = [col for col in self.features_df.columns if col not in exclude_columns]
        
        # Prepare features and labels
        self.X = self.features_df[feature_columns]
        self.y = self.features_df['task_type']
        
        # Handle missing values
        self.X = self.X.fillna(0)
        
        # Remove constant features
        self.X = self.X.loc[:, self.X.std() > 0]
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f" Data preprocessed:")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Samples: {self.X.shape[0]}")
        print(f"   Classes: {self.y.unique()}")
        
        return True
    
    def feature_selection(self, n_features=50):
        """Select most important features"""
        if self.X_scaled is None:
            print(" Data not preprocessed")
            return None
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_scaled, self.y)
        
        # Get feature importance
        importance_scores = rf.feature_importances_
        feature_names = self.X.columns
        
        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = self.feature_importance.head(n_features)['feature'].values
        feature_mask = [col in selected_features for col in self.X.columns]
        
        self.X_selected = self.X_scaled[:, feature_mask]
        self.selected_feature_names = selected_features
        
        print(f" Selected top {n_features} features")
        print("Top 10 features:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return self.X_selected
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Parameter grids for hyperparameter tuning
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def evaluate_model(self, model, X, y, model_name, cv_folds=5):
        """Comprehensive model evaluation with cross-validation"""
        print(f"\n Evaluating {model_name}...")
        
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        # Additional metrics using cross_val_predict
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(model, X, y, cv=skf)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        results = {
            'model': model,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores
        }
        
        print(f"   Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        return results
    
    def train_and_evaluate_all_models(self, use_feature_selection=True, cv_folds=5):
        """Train and evaluate all models"""
        if self.X_scaled is None:
            print(" Data not preprocessed")
            return
        
        self.initialize_models()
        
        # Use selected features or all features
        if use_feature_selection and self.X.shape[1] > 50:
            X_used = self.X_selected
            print(f"Using {X_used.shape[1]} selected features")
        else:
            X_used = self.X_scaled
            print(f"Using all {X_used.shape[1]} features")
        
        self.results = {}
        
        for model_name, model in self.models.items():
            # Simple evaluation first
            results = self.evaluate_model(model, X_used, self.y, model_name, cv_folds)
            self.results[model_name] = results
            
            # Hyperparameter tuning for best models
            if model_name in ['Random Forest', 'SVM']:
                print(f"    Tuning hyperparameters for {model_name}...")
                grid_search = GridSearchCV(
                    model, self.param_grids[model_name], 
                    cv=cv_folds, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_used, self.y)
                
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                
                print(f"   Best parameters: {grid_search.best_params_}")
                print(f"   Best CV accuracy: {best_score:.3f}")
                
                # Update with tuned model
                self.results[model_name]['tuned_model'] = best_model
                self.results[model_name]['best_score'] = best_score
        
        return self.results
    
    def compare_models(self):
        """Compare performance of all models"""
        if not self.results:
            print(" No results to compare")
            return
        
        print("\n" + "="*60)
        print(" MODEL COMPARISON SUMMARY")
        print("="*60)
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'CV Accuracy': f"{result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f}",
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['cv_accuracy_mean'])
        best_result = self.results[best_model_name]
        
        print(f"\n BEST MODEL: {best_model_name}")
        print(f"   Cross-validation Accuracy: {best_result['cv_accuracy_mean']:.3f}")
        print(f"   F1-Score: {best_result['f1_score']:.3f}")
        
        return best_model_name, best_result
    
    def plot_results(self):
        """Create comprehensive visualization of results"""
        if not self.results:
            print(" No results to visualize")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Plot
        model_names = list(self.results.keys())
        cv_scores = [self.results[name]['cv_accuracy_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_accuracy_std'] for name in model_names]
        
        bars = ax1.bar(model_names, cv_scores, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_title('Model Comparison - Cross Validation Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, cv_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature Importance (for Random Forest)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            ax2.barh(range(len(top_features)), top_features['importance'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Importance Score')
        
        # 3. Confusion Matrix for Best Model
        best_model_name, best_result = self.compare_models()
        best_model = best_result['model']
        
        # Use cross_val_predict for confusion matrix
        from sklearn.model_selection import cross_val_predict
        if self.X_selected is not None:
            X_used = self.X_selected
        else:
            X_used = self.X_scaled
            
        y_pred = cross_val_predict(best_model, X_used, self.y, cv=5)
        
        cm = confusion_matrix(self.y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=sorted(self.y.unique()), 
                   yticklabels=sorted(self.y.unique()))
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Detailed Metrics Comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            result = self.results[model_name]
            metric_data['Precision'].append(result['precision'])
            metric_data['Recall'].append(result['recall'])
            metric_data['F1-Score'].append(result['f1_score'])
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (metric, values) in enumerate(metric_data.items()):
            ax4.bar(x + i*width, values, width, label=metric, alpha=0.7)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Score')
        ax4.set_title('Detailed Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Visualizations saved as 'model_comparison_results.png'")
    
    def run_complete_analysis(self, features_file='eeg_features.csv', n_features=50, cv_folds=5):
        """Run complete classification pipeline"""
        print(" MENTAL STATE CLASSIFICATION ANALYSIS")
        print("="*60)
        
        # Load features
        if not self.load_features(features_file):
            return
        
        # Preprocess data
        if not self.preprocess_data():
            return
        
        # Feature selection
        self.feature_selection(n_features=n_features)
        
        # Train and evaluate models
        self.train_and_evaluate_all_models(use_feature_selection=True, cv_folds=cv_folds)
        
        # Compare results
        best_model, best_result = self.compare_models()
        
        # Create visualizations
        self.plot_results()
        
        # Final report
        print("\n" + "="*60)
        print(" FINAL CLASSIFICATION REPORT")
        print("="*60)
        print(f"Dataset: {self.X.shape[0]} samples, {self.X_selected.shape[1]} features")
        print(f"Best Model: {best_model}")
        print(f"Cross-validation Accuracy: {best_result['cv_accuracy_mean']:.3f} ± {best_result['cv_accuracy_std']:.3f}")
        print(f"F1-Score: {best_result['f1_score']:.3f}")
        
        return best_model, best_result

def main():
    """Main execution function"""
    classifier = MentalStateClassifier()
    
    # Run complete analysis
    best_model, best_result = classifier.run_complete_analysis(
        features_file='eeg_features.csv',
        n_features=30,  # Use top 30 features
        cv_folds=5      # 5-fold cross validation
    )

if __name__ == "__main__":
    main()