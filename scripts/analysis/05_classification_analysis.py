"""
StAMA - Statistical Analysis in Modern Academia
Script 5: Classification Analysis

This script develops and evaluates classification models to predict journal legitimacy
and identify patterns that may indicate publication misconduct.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare data for classification analysis"""
    print("Loading and preparing data for classification analysis...")
    df = pd.read_csv('../../data/cleaned/Data_with_Labels.csv')
    
    # Select features for classification
    classification_features = [
        'SJR', 'H index', 'IPP', 'SNIP',
        'Self-Cites/Total Cites (3years)',
        'Uncited Docs./Total Docs. (3years)',
        'Cites / Doc. (2years)',
        'Ref. / Doc.',
        'Total Docs. (2023)',
        'Total Cites (3years)',
        'Coverage_Duration'
    ]
    
    # Create feature matrix and target vector
    X = df[classification_features].copy()
    y = df['Labels'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Convert labels to binary (1 for legitimate, 0 for problematic)
    y = (y == 1.0).astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:")
    print(f"Legitimate journals (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"Problematic journals (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    return X, y, classification_features

def feature_selection_analysis(X, y, feature_names):
    """Perform feature selection analysis"""
    print("\n" + "="*60)
    print("FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # Univariate feature selection
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values('Score', ascending=False)
    
    print("Univariate Feature Selection (F-statistic):")
    print(feature_scores)
    
    # Save feature scores
    feature_scores.to_csv('images/feature_selection_scores.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feature_scores)), feature_scores['Score'])
    plt.yticks(range(len(feature_scores)), feature_scores['Feature'])
    plt.xlabel('F-statistic Score')
    plt.title('Feature Importance (Univariate F-test)')
    plt.grid(True, alpha=0.3)
    
    # Color bars based on p-value significance
    for i, (bar, p_val) in enumerate(zip(bars, feature_scores['P_value'])):
        if p_val < 0.001:
            bar.set_color('red')
        elif p_val < 0.01:
            bar.set_color('orange')
        elif p_val < 0.05:
            bar.set_color('yellow')
        else:
            bar.set_color('gray')
    
    plt.tight_layout()
    plt.savefig('images/feature_importance_univariate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_scores

def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classification models"""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    # Store results
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            auc_score = np.nan
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC': auc_score,
            'CV_Mean': cv_mean,
            'CV_Std': cv_std
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('images/classification_results.csv', index=False)
    
    return results_df, trained_models

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for best models"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Random Forest hyperparameter tuning
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    
    print(f"Best RF parameters: {rf_grid.best_params_}")
    print(f"Best RF score: {rf_grid.best_score_:.4f}")
    
    # Gradient Boosting hyperparameter tuning
    print("\nTuning Gradient Boosting...")
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    gb_grid.fit(X_train, y_train)
    
    print(f"Best GB parameters: {gb_grid.best_params_}")
    print(f"Best GB score: {gb_grid.best_score_:.4f}")
    
    # Save tuning results
    tuning_results = {
        'Random_Forest': {
            'best_params': rf_grid.best_params_,
            'best_score': rf_grid.best_score_
        },
        'Gradient_Boosting': {
            'best_params': gb_grid.best_params_,
            'best_score': gb_grid.best_score_
        }
    }
    
    # Save as DataFrame
    tuning_df = pd.DataFrame([
        {'Model': 'Random Forest', 'Best_Score': rf_grid.best_score_, 'Best_Params': str(rf_grid.best_params_)},
        {'Model': 'Gradient Boosting', 'Best_Score': gb_grid.best_score_, 'Best_Params': str(gb_grid.best_params_)}
    ])
    tuning_df.to_csv('images/hyperparameter_tuning_results.csv', index=False)
    
    return rf_grid.best_estimator_, gb_grid.best_estimator_

def create_model_comparison_plots(results_df):
    """Create comparison plots for different models"""
    print("Creating model comparison visualizations...")
    
    # Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(results_df['Model'], results_df['Accuracy'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1-score comparison
    axes[0, 1].bar(results_df['Model'], results_df['F1_Score'])
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC comparison
    axes[1, 0].bar(results_df['Model'], results_df['AUC'])
    axes[1, 0].set_title('Model AUC Comparison')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cross-validation scores
    axes[1, 1].bar(results_df['Model'], results_df['CV_Mean'])
    axes[1, 1].errorbar(range(len(results_df)), results_df['CV_Mean'], 
                       yerr=results_df['CV_Std'], fmt='none', color='red', capsize=5)
    axes[1, 1].set_title('Cross-Validation Accuracy')
    axes[1, 1].set_ylabel('CV Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/model_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(trained_models, X_test, y_test):
    """Create ROC curves for all models"""
    print("Creating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrices(trained_models, X_test, y_test):
    """Create confusion matrices for all models"""
    print("Creating confusion matrices...")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Remove empty subplot
    if n_models < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def feature_importance_analysis(best_model, feature_names):
    """Analyze feature importance from the best model"""
    print("Analyzing feature importance...")
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance (from best model):")
        print(feature_importance_df)
        
        # Save feature importance
        feature_importance_df.to_csv('images/feature_importance_best_model.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance from Best Model')
        plt.grid(True, alpha=0.3)
        
        # Color bars based on importance
        max_importance = feature_importance_df['Importance'].max()
        for bar, importance in zip(bars, feature_importance_df['Importance']):
            if importance > 0.7 * max_importance:
                bar.set_color('red')
            elif importance > 0.4 * max_importance:
                bar.set_color('orange')
            elif importance > 0.2 * max_importance:
                bar.set_color('yellow')
            else:
                bar.set_color('lightblue')
        
        plt.tight_layout()
        plt.savefig('images/feature_importance_best_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance_df
    else:
        print("Best model does not have feature_importances_ attribute")
        return None

def predict_suspicious_journals(best_model, X, y, feature_names):
    """Use the best model to identify potentially suspicious journals"""
    print("Identifying potentially suspicious journals...")
    
    # Get predictions and probabilities
    y_pred = best_model.predict(X)
    y_pred_proba = best_model.predict_proba(X)[:, 0]  # Probability of being problematic
    
    # Create results DataFrame
    df = pd.read_csv('Data_with_Labels.csv')

    # Filter to match our processed data
    df_filtered = df.dropna(subset=['Labels'])

    # Only fill numeric columns with median
    numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
    df_filtered[numeric_columns] = df_filtered[numeric_columns].fillna(df_filtered[numeric_columns].median())
    
    results_df = df_filtered[['Title', 'Publisher', 'Labels']].copy()
    results_df['Predicted_Label'] = y_pred
    results_df['Problematic_Probability'] = y_pred_proba
    results_df['True_Positive'] = (results_df['Labels'] == 0.0) & (results_df['Predicted_Label'] == 0)
    results_df['False_Positive'] = (results_df['Labels'] == 1.0) & (results_df['Predicted_Label'] == 0)
    
    # Sort by problematic probability
    results_df = results_df.sort_values('Problematic_Probability', ascending=False)
    
    # Save results
    results_df.to_csv('images/journal_predictions.csv', index=False)
    
    # Print top suspicious journals
    print("\nTop 10 journals flagged as potentially problematic:")
    print(results_df.head(10)[['Title', 'Publisher', 'Problematic_Probability', 'Labels']])
    
    return results_df

def main():
    """Main function to run classification analysis"""
    # Create images directory
    import os
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 5: Classification Analysis")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection analysis
    feature_scores = feature_selection_analysis(X_train_scaled, y_train, feature_names)
    
    # Train models
    results_df, trained_models = train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Hyperparameter tuning
    best_rf, best_gb = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Evaluate best models
    best_model = best_rf if results_df[results_df['Model'] == 'Random Forest']['F1_Score'].iloc[0] > \
                           results_df[results_df['Model'] == 'Gradient Boosting']['F1_Score'].iloc[0] else best_gb
    
    # Create visualizations
    create_model_comparison_plots(results_df)
    create_roc_curves(trained_models, X_test_scaled, y_test)
    create_confusion_matrices(trained_models, X_test_scaled, y_test)
    
    # Feature importance analysis
    feature_importance_df = feature_importance_analysis(best_model, feature_names)
    
    # Predict suspicious journals
    X_scaled = scaler.fit_transform(X)
    predictions_df = predict_suspicious_journals(best_model, X_scaled, y, feature_names)
    
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved in 'images' directory:")
    print("- feature_selection_scores.csv")
    print("- feature_importance_univariate.png")
    print("- classification_results.csv")
    print("- hyperparameter_tuning_results.csv")
    print("- model_comparison_plots.png")
    print("- roc_curves_comparison.png")
    print("- confusion_matrices.png")
    print("- feature_importance_best_model.csv and .png")
    print("- journal_predictions.csv")

if __name__ == "__main__":
    main()
