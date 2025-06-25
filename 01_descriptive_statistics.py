"""
StAMA - Statistical Analysis in Modern Academia
Script 1: Descriptive Statistics Analysis

This script performs comprehensive descriptive statistical analysis of journal data
to identify patterns that may indicate publication misconduct.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the journal data for analysis"""
    print("Loading journal data...")
    df = pd.read_csv(r'C:\Users\TEJA\Downloads\tj\tj\Data_with_Labels.csv')
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Number of legitimate journals (Label=1): {sum(df['Labels'] == 1.0)}")
    print(f"Number of potentially problematic journals (Label=0): {sum(df['Labels'] == 0.0)}")
    
    return df

def basic_descriptive_stats(df):
    """Generate basic descriptive statistics"""
    print("\n" + "="*60)
    print("BASIC DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Select numerical columns for analysis
    numerical_cols = ['SJR', 'H index', 'IPP', 'SNIP', 'Total Docs. (2023)', 
                     'Total Docs. (3years)', 'Total Refs.', 'Est. value (USD) (2023)',
                     'Total Cites (3years)', 'Self-Cites/Total Cites (3years)',
                     'Uncited Docs./Total Docs. (3years)', 'Citable Docs. (3years)',
                     'Cites / Doc. (2years)', 'Ref. / Doc.', 'Coverage_Duration']
    
    # Overall descriptive statistics
    desc_stats = df[numerical_cols].describe()
    print("\nOverall Descriptive Statistics:")
    print(desc_stats.round(3))
    
    # Save to CSV
    desc_stats.to_csv('images/descriptive_statistics_overall.csv')
    
    # Descriptive statistics by label
    print("\nDescriptive Statistics by Journal Type:")
    for label in [1.0, 0.0]:
        label_name = "Legitimate" if label == 1.0 else "Potentially Problematic"
        print(f"\n{label_name} Journals:")
        subset = df[df['Labels'] == label][numerical_cols]
        desc_subset = subset.describe()
        print(desc_subset.round(3))
        
        # Save to CSV
        filename = f'images/descriptive_statistics_{label_name.lower().replace(" ", "_")}.csv'
        desc_subset.to_csv(filename)
    
    return desc_stats

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("\n" + "="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print("Missing Data Summary:")
        print(missing_df)
        missing_df.to_csv('images/missing_data_analysis.csv')
    else:
        print("No missing data found in the dataset.")

def distribution_analysis(df):
    """Analyze distributions of key variables"""
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Key metrics for distribution analysis
    key_metrics = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                   'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)']
    
    # Create distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(key_metrics):
        if i < len(axes):
            # Histogram with KDE
            axes[i].hist(df[metric], bins=50, alpha=0.7, density=True, color='skyblue')
            
            # Add KDE
            try:
                kde_data = df[metric].dropna()
                if len(kde_data) > 1:
                    kde = stats.gaussian_kde(kde_data)
                    x_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2)
            except:
                pass
            
            axes[i].set_title(f'Distribution of {metric}')
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel('Density')
    
    # Remove empty subplots
    for i in range(len(key_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('images/distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical tests for normality
    print("\nNormality Tests (Shapiro-Wilk p-values):")
    normality_results = {}
    for metric in key_metrics:
        data = df[metric].dropna()
        if len(data) > 3:  # Minimum sample size for Shapiro-Wilk
            # Use a sample if data is too large
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            stat, p_value = stats.shapiro(data)
            normality_results[metric] = p_value
            print(f"{metric}: p = {p_value:.6f}")
    
    # Save normality test results
    normality_df = pd.DataFrame(list(normality_results.items()), 
                               columns=['Metric', 'Shapiro_Wilk_p_value'])
    normality_df.to_csv('images/normality_tests.csv', index=False)

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numerical columns for correlation
    numerical_cols = ['SJR', 'H index', 'IPP', 'SNIP', 'Total Docs. (2023)', 
                     'Total Docs. (3years)', 'Total Refs.', 'Total Cites (3years)',
                     'Self-Cites/Total Cites (3years)', 'Uncited Docs./Total Docs. (3years)',
                     'Citable Docs. (3years)', 'Cites / Doc. (2years)', 'Ref. / Doc.',
                     'Coverage_Duration', 'Labels']
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Matrix of Journal Metrics')
    plt.tight_layout()
    plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation matrix
    corr_matrix.to_csv('images/correlation_matrix.csv')
    
    # Find strongest correlations with Labels
    label_correlations = corr_matrix['Labels'].abs().sort_values(ascending=False)
    print("\nStrongest correlations with journal legitimacy (Labels):")
    print(label_correlations.head(10))
    
    return corr_matrix

def outlier_analysis(df):
    """Identify and analyze outliers"""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)
    
    key_metrics = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                   'Uncited Docs./Total Docs. (3years)']
    
    outlier_summary = {}
    
    for metric in key_metrics:
        data = df[metric].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        outlier_summary[metric] = {
            'Count': outlier_count,
            'Percentage': outlier_percentage,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        }
        
        print(f"{metric}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
    
    # Save outlier analysis
    outlier_df = pd.DataFrame(outlier_summary).T
    outlier_df.to_csv('images/outlier_analysis.csv')
    
    return outlier_summary

def main():
    """Main function to run all descriptive analyses"""
    # Create images directory
    import os
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 1: Descriptive Statistics Analysis")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    desc_stats = basic_descriptive_stats(df)
    analyze_missing_data(df)
    distribution_analysis(df)
    corr_matrix = correlation_analysis(df)
    outlier_summary = outlier_analysis(df)
    
    print("\n" + "="*60)
    print("DESCRIPTIVE ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved in 'images' directory:")
    print("- descriptive_statistics_*.csv")
    print("- distribution_analysis.png")
    print("- correlation_matrix.png and .csv")
    print("- normality_tests.csv")
    print("- outlier_analysis.csv")
    print("- missing_data_analysis.csv (if applicable)")

if __name__ == "__main__":
    main()
