"""
StAMA - Statistical Analysis in Modern Academia
Script 2: Data Visualization

This script creates comprehensive visualizations to explore patterns in journal data
and identify potential indicators of publication misconduct.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the journal data"""
    print("Loading journal data for visualization...")
    df = pd.read_csv('Data_with_Labels.csv')
    return df

def create_comparison_plots(df):
    """Create comparison plots between legitimate and problematic journals"""
    print("Creating comparison visualizations...")
    
    # Key metrics for comparison
    key_metrics = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                   'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)', 'Ref. / Doc.']
    
    # Create box plots comparing legitimate vs problematic journals
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(key_metrics):
        # Prepare data for plotting
        legitimate = df[df['Labels'] == 1.0][metric].dropna()
        problematic = df[df['Labels'] == 0.0][metric].dropna()
        
        # Box plot
        data_to_plot = [legitimate, problematic]
        labels = ['Legitimate', 'Problematic']
        
        box_plot = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[i].set_title(f'{metric}\nComparison by Journal Type')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/journal_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_comparisons(df):
    """Create distribution comparison plots"""
    print("Creating distribution comparison plots...")
    
    key_metrics = ['SJR', 'H index', 'Self-Cites/Total Cites (3years)', 'Uncited Docs./Total Docs. (3years)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(key_metrics):
        legitimate = df[df['Labels'] == 1.0][metric].dropna()
        problematic = df[df['Labels'] == 0.0][metric].dropna()
        
        # Create histograms
        axes[i].hist(legitimate, bins=50, alpha=0.7, label='Legitimate', 
                    color='lightblue', density=True)
        axes[i].hist(problematic, bins=50, alpha=0.7, label='Problematic', 
                    color='lightcoral', density=True)
        
        axes[i].set_title(f'Distribution of {metric}')
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/distribution_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_scatter_plots(df):
    """Create scatter plots to identify patterns"""
    print("Creating scatter plot analysis...")
    
    # Key relationships to explore
    relationships = [
        ('SJR', 'H index'),
        ('Self-Cites/Total Cites (3years)', 'SJR'),
        ('Uncited Docs./Total Docs. (3years)', 'H index'),
        ('IPP', 'SNIP')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (x_metric, y_metric) in enumerate(relationships):
        # Separate data by label
        legitimate = df[df['Labels'] == 1.0]
        problematic = df[df['Labels'] == 0.0]
        
        # Create scatter plots
        axes[i].scatter(legitimate[x_metric], legitimate[y_metric], 
                       alpha=0.6, c='blue', label='Legitimate', s=20)
        axes[i].scatter(problematic[x_metric], problematic[y_metric], 
                       alpha=0.6, c='red', label='Problematic', s=20)
        
        axes[i].set_xlabel(x_metric)
        axes[i].set_ylabel(y_metric)
        axes[i].set_title(f'{y_metric} vs {x_metric}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/scatter_plot_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_publisher_analysis(df):
    """Analyze patterns by publisher"""
    print("Creating publisher analysis...")
    
    # Count journals by publisher and label
    publisher_counts = df.groupby(['Publisher', 'Labels']).size().unstack(fill_value=0)
    
    # Calculate total journals per publisher
    publisher_counts['Total'] = publisher_counts.sum(axis=1)
    
    # Calculate percentage of problematic journals
    if 0.0 in publisher_counts.columns:
        publisher_counts['Problematic_Percentage'] = (publisher_counts[0.0] / publisher_counts['Total']) * 100
    else:
        publisher_counts['Problematic_Percentage'] = 0
    
    # Filter publishers with at least 5 journals
    significant_publishers = publisher_counts[publisher_counts['Total'] >= 5]
    
    # Sort by problematic percentage
    significant_publishers = significant_publishers.sort_values('Problematic_Percentage', ascending=False)
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    top_publishers = significant_publishers.head(20)
    
    bars = plt.bar(range(len(top_publishers)), top_publishers['Problematic_Percentage'])
    plt.xlabel('Publishers')
    plt.ylabel('Percentage of Problematic Journals')
    plt.title('Publishers with Highest Percentage of Potentially Problematic Journals\n(Publishers with ≥5 journals)')
    plt.xticks(range(len(top_publishers)), top_publishers.index, rotation=45, ha='right')
    
    # Color bars based on percentage
    for i, bar in enumerate(bars):
        percentage = top_publishers.iloc[i]['Problematic_Percentage']
        if percentage > 50:
            bar.set_color('red')
        elif percentage > 25:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/publisher_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save publisher analysis data
    significant_publishers.to_csv('images/publisher_analysis.csv')

def create_citation_patterns(df):
    """Analyze citation patterns that might indicate misconduct"""
    print("Analyzing citation patterns...")
    
    # Create a comprehensive citation analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Self-citation rate distribution
    axes[0, 0].hist(df[df['Labels'] == 1.0]['Self-Cites/Total Cites (3years)'], 
                   bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[0, 0].hist(df[df['Labels'] == 0.0]['Self-Cites/Total Cites (3years)'], 
                   bins=50, alpha=0.7, label='Problematic', color='red', density=True)
    axes[0, 0].set_title('Self-Citation Rate Distribution')
    axes[0, 0].set_xlabel('Self-Cites/Total Cites')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Uncited documents ratio
    axes[0, 1].hist(df[df['Labels'] == 1.0]['Uncited Docs./Total Docs. (3years)'], 
                   bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[0, 1].hist(df[df['Labels'] == 0.0]['Uncited Docs./Total Docs. (3years)'], 
                   bins=50, alpha=0.7, label='Problematic', color='red', density=True)
    axes[0, 1].set_title('Uncited Documents Ratio Distribution')
    axes[0, 1].set_xlabel('Uncited Docs./Total Docs.')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Citations per document
    axes[0, 2].hist(df[df['Labels'] == 1.0]['Cites / Doc. (2years)'], 
                   bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[0, 2].hist(df[df['Labels'] == 0.0]['Cites / Doc. (2years)'], 
                   bins=50, alpha=0.7, label='Problematic', color='red', density=True)
    axes[0, 2].set_title('Citations per Document Distribution')
    axes[0, 2].set_xlabel('Cites / Doc. (2years)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Self-citation vs SJR
    legitimate = df[df['Labels'] == 1.0]
    problematic = df[df['Labels'] == 0.0]
    
    axes[1, 0].scatter(legitimate['Self-Cites/Total Cites (3years)'], legitimate['SJR'], 
                      alpha=0.6, c='blue', label='Legitimate', s=20)
    axes[1, 0].scatter(problematic['Self-Cites/Total Cites (3years)'], problematic['SJR'], 
                      alpha=0.6, c='red', label='Problematic', s=20)
    axes[1, 0].set_xlabel('Self-Cites/Total Cites')
    axes[1, 0].set_ylabel('SJR')
    axes[1, 0].set_title('Self-Citation Rate vs Journal Ranking')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Uncited docs vs H-index
    axes[1, 1].scatter(legitimate['Uncited Docs./Total Docs. (3years)'], legitimate['H index'], 
                      alpha=0.6, c='blue', label='Legitimate', s=20)
    axes[1, 1].scatter(problematic['Uncited Docs./Total Docs. (3years)'], problematic['H index'], 
                      alpha=0.6, c='red', label='Problematic', s=20)
    axes[1, 1].set_xlabel('Uncited Docs./Total Docs.')
    axes[1, 1].set_ylabel('H index')
    axes[1, 1].set_title('Uncited Documents vs H-index')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. References per document distribution
    axes[1, 2].hist(df[df['Labels'] == 1.0]['Ref. / Doc.'], 
                   bins=50, alpha=0.7, label='Legitimate', color='blue', density=True)
    axes[1, 2].hist(df[df['Labels'] == 0.0]['Ref. / Doc.'], 
                   bins=50, alpha=0.7, label='Problematic', color='red', density=True)
    axes[1, 2].set_title('References per Document Distribution')
    axes[1, 2].set_xlabel('Ref. / Doc.')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/citation_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_plots(df):
    """Create interactive plots using Plotly"""
    print("Creating interactive visualizations...")
    
    # Interactive scatter plot
    fig = px.scatter(df, x='SJR', y='H index', color='Labels', 
                     hover_data=['Title', 'Publisher', 'Self-Cites/Total Cites (3years)'],
                     title='Interactive Journal Analysis: SJR vs H-index',
                     labels={'Labels': 'Journal Type'})
    
    # Update color scale
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.write_html('images/interactive_scatter_plot.html')
    
    # Interactive correlation heatmap
    numerical_cols = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                     'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)', 'Labels']
    
    corr_matrix = df[numerical_cols].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False))
    
    fig_heatmap.update_layout(
        title='Interactive Correlation Matrix',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    
    fig_heatmap.write_html('images/interactive_correlation_heatmap.html')

def main():
    """Main function to run all visualization analyses"""
    # Create images directory
    import os
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 2: Data Visualization")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    create_comparison_plots(df)
    create_distribution_comparisons(df)
    create_scatter_plots(df)
    create_publisher_analysis(df)
    create_citation_patterns(df)
    create_interactive_plots(df)
    
    print("\n" + "="*60)
    print("VISUALIZATION ANALYSIS COMPLETE")
    print("="*60)
    print("Visualizations saved in 'images' directory:")
    print("- journal_comparison_boxplots.png")
    print("- distribution_comparisons.png")
    print("- scatter_plot_analysis.png")
    print("- publisher_analysis.png and .csv")
    print("- citation_patterns_analysis.png")
    print("- interactive_scatter_plot.html")
    print("- interactive_correlation_heatmap.html")

if __name__ == "__main__":
    main()
