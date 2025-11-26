import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import warnings
warnings.filterwarnings('ignore')


class SAM40BinaryClassifier:
    """
    Binary classifier for SAM40 dataset:
    MATH (arithmetic task) vs RELAX (rest/relaxation).
    Uses precomputed features from sam40_eeg_features.csv.
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
        self.X_selected = None
        self.selected_feature_names = None

    # -------------------------------------------------------
    # Load features
    # -------------------------------------------------------
    def load_features(self, features_file='sam40_eeg_features.csv'):
        """Load the extracted SAM40 features and filter to MATH vs RELAX."""
        try:
            df = pd.read_csv(features_file)
            print(f"\nLoaded feature file: {features_file}")
            print(f"Shape: {df.shape}")

            # Filter to MATH and RELAX only
            df = df[df['task_type'].isin(['MATH', 'RELAX'])].copy()
            if df.empty:
                print("No MATH/RELAX samples found in the feature file.")
                return False

            print("\nTask distribution (after filtering to MATH & RELAX):")
            print(df['task_type'].value_counts())

            self.features_df = df
            return True

        except FileNotFoundError:
            print(f"Features file '{features_file}' not found.")
            return False

    # -------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------
    def preprocess_data(self):
        """Preprocess data for machine learning."""
        if self.features_df is None:
            print("No features loaded.")
            return False

        # Non-feature metadata columns
        exclude_columns = [
            'file_name', 'task_type', 'subject_id',
            'duration', 'num_channels', 'sampling_rate'
        ]
        feature_columns = [
            col for col in self.features_df.columns
            if col not in exclude_columns
        ]

        # Features and labels
        self.X = self.features_df[feature_columns].copy()
        self.y = self.features_df['task_type'].copy()

        # Handle missing values
        self.X = self.X.fillna(0)

        # Remove constant features
        stds = self.X.std()
        self.X = self.X.loc[:, stds > 0]

        # Scale
        self.X_scaled = self.scaler.fit_transform(self.X)

        print("\nData preprocessed:")
        print(f"  Samples:  {self.X.shape[0]}")
        print(f"  Features: {self.X.shape[1]}")
        print(f"  Classes:  {self.y.unique()}")
        return True

    # -------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------
    def feature_selection(self, n_features=100):
        """Select most important features using Random Forest importance."""
        if self.X_scaled is None:
            print("Data not preprocessed.")
            return None

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_scaled, self.y)

        importance_scores = rf.feature_importances_
        feature_names = self.X.columns

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        # Select top n_features (but not more than available)
        n_features = min(n_features, len(feature_names))
        selected_features = self.feature_importance.head(n_features)['feature'].values
        feature_mask = [col in selected_features for col in self.X.columns]

        self.X_selected = self.X_scaled[:, feature_mask]
        self.selected_feature_names = selected_features

        print(f"\nSelected top {n_features} features.")
        print("Top 10 features:")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return self.X_selected

    # -------------------------------------------------------
    # Model setup
    # -------------------------------------------------------
    def initialize_models(self):
        """Initialize multiple ML models for binary classification."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=1000
            )
        }

        self.param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 0.01, 0.001]
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

    # -------------------------------------------------------
    # Evaluation helper
    # -------------------------------------------------------
    def evaluate_model(self, model, X, y, model_name, cv_folds=5):
        """Comprehensive model evaluation with cross-validation."""
        print(f"\nEvaluating {model_name}...")

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        cv_scores = cross_val_score(
            model, X, y, cv=skf, scoring='accuracy'
        )

        y_pred = cross_val_predict(model, X, y, cv=skf)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print(f"  Precision:   {precision:.3f}")
        print(f"  Recall:      {recall:.3f}")
        print(f"  F1-score:    {f1:.3f}")

        return {
            'model': model,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores
        }

    # -------------------------------------------------------
    # Train all models
    # -------------------------------------------------------
    def train_and_evaluate_all_models(self, use_feature_selection=True, cv_folds=5):
        if self.X_scaled is None:
            print("Data not preprocessed.")
            return

        self.initialize_models()

        if use_feature_selection and self.X_selected is not None:
            X_used = self.X_selected
            print(f"\nUsing {X_used.shape[1]} selected features.")
        else:
            X_used = self.X_scaled
            print(f"\nUsing all {X_used.shape[1]} features (no selection).")

        self.results = {}

        for model_name, model in self.models.items():
            res = self.evaluate_model(model, X_used, self.y, model_name, cv_folds)
            self.results[model_name] = res

            # Hyperparameter tuning for RF & SVM only (to save time)
            if model_name in ['Random Forest', 'SVM']:
                print(f"  Tuning hyperparameters for {model_name}...")
                grid = GridSearchCV(
                    model,
                    self.param_grids[model_name],
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid.fit(X_used, self.y)

                best_model = grid.best_estimator_
                best_score = grid.best_score_

                print(f"   Best params: {grid.best_params_}")
                print(f"   Best CV accuracy: {best_score:.3f}")

                self.results[model_name]['tuned_model'] = best_model
                self.results[model_name]['best_score'] = best_score

        return self.results

    # -------------------------------------------------------
    # Comparison
    # -------------------------------------------------------
    def compare_models(self):
        if not self.results:
            print("No results to compare.")
            return

        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        rows = []
        for name, res in self.results.items():
            rows.append({
                'Model': name,
                'CV_Acc': f"{res['cv_accuracy_mean']:.3f} ± {res['cv_accuracy_std']:.3f}",
                'Precision': f"{res['precision']:.3f}",
                'Recall': f"{res['recall']:.3f}",
                'F1': f"{res['f1_score']:.3f}"
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        best_name = max(self.results.keys(),
                        key=lambda k: self.results[k]['cv_accuracy_mean'])
        best_res = self.results[best_name]

        print(f"\nBest model: {best_name}")
        print(f"  CV accuracy: {best_res['cv_accuracy_mean']:.3f}")
        print(f"  F1-score:    {best_res['f1_score']:.3f}")

        return best_name, best_res

    # -------------------------------------------------------
    # Confusion matrix plot
    # -------------------------------------------------------
    def plot_confusion_for_best(self, cv_folds=5):
        if not self.results:
            print("No results to visualize.")
            return

        best_name, best_res = self.compare_models()
        best_model = best_res['model']

        if self.X_selected is not None:
            X_used = self.X_selected
        else:
            X_used = self.X_scaled

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )
        y_pred = cross_val_predict(best_model, X_used, self.y, cv=skf)

        cm = confusion_matrix(self.y, y_pred, labels=sorted(self.y.unique()))

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(self.y.unique()),
            yticklabels=sorted(self.y.unique())
        )
        plt.title(f'Confusion Matrix – {best_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------
    def run_complete_analysis(self, features_file='sam40_eeg_features.csv',
                              n_features=100, cv_folds=5):
        print("\nSAM40 MATH vs RELAX CLASSIFICATION")
        print("=" * 60)

        if not self.load_features(features_file):
            return

        if not self.preprocess_data():
            return

        self.feature_selection(n_features=n_features)
        self.train_and_evaluate_all_models(
            use_feature_selection=True,
            cv_folds=cv_folds
        )
        best_name, best_res = self.compare_models()
        self.plot_confusion_for_best(cv_folds=cv_folds)

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Samples: {self.X.shape[0]}")
        print(f"Features used: {self.X_selected.shape[1]}")
        print(f"Best model: {best_name}")
        print(f"Best CV accuracy: {best_res['cv_accuracy_mean']:.3f}")
        print(f"Best F1-score: {best_res['f1_score']:.3f}")

        return best_name, best_res


def main():
    clf = SAM40BinaryClassifier()
    clf.run_complete_analysis(
        features_file='sam40_eeg_features.csv',
        n_features=100,
        cv_folds=5
    )


if __name__ == "__main__":
    main()
