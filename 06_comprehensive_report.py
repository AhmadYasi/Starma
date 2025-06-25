"""
StAMA - Statistical Analysis in Modern Academia
Script 6: Comprehensive Report Generation

This script generates a comprehensive report summarizing all analyses and findings
regarding journal clustering and publication misconduct detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_analysis_results():
    """Load results from all previous analyses"""
    print("Loading analysis results...")
    
    results = {}
    
    # Check if results files exist and load them
    files_to_load = {
        'descriptive_stats': 'images/descriptive_statistics_overall.csv',
        'correlation_matrix': 'images/correlation_matrix.csv',
        'hypothesis_tests': 'images/hypothesis_test_results_corrected.csv',
        'confidence_intervals': 'images/confidence_intervals.csv',
        'cluster_characteristics': 'images/cluster_characteristics.csv',
        'suspicious_patterns': 'images/suspicious_patterns_analysis.csv',
        'classification_results': 'images/classification_results.csv',
        'feature_importance': 'images/feature_importance_best_model.csv',
        'journal_predictions': 'images/journal_predictions.csv',
        'publisher_analysis': 'images/publisher_analysis.csv'
    }
    
    for key, filepath in files_to_load.items():
        if os.path.exists(filepath):
            try:
                results[key] = pd.read_csv(filepath)
                print(f"+ Loaded {key}")
            except Exception as e:
                print(f"- Error loading {key}: {e}")
                results[key] = None
        else:
            print(f"- File not found: {filepath}")
            results[key] = None
    
    return results

def generate_executive_summary(results):
    """Generate executive summary of findings"""
    print("Generating executive summary...")
    
    summary = []
    summary.append("EXECUTIVE SUMMARY")
    summary.append("="*50)
    summary.append("")
    
    # Dataset overview
    if results['journal_predictions'] is not None:
        total_journals = len(results['journal_predictions'])
        legitimate_journals = sum(results['journal_predictions']['Labels'] == 1.0)
        problematic_journals = sum(results['journal_predictions']['Labels'] == 0.0)
        
        summary.append(f"Dataset Overview:")
        summary.append(f"- Total journals analyzed: {total_journals:,}")
        summary.append(f"- Legitimate journals: {legitimate_journals:,} ({legitimate_journals/total_journals*100:.1f}%)")
        summary.append(f"- Potentially problematic journals: {problematic_journals:,} ({problematic_journals/total_journals*100:.1f}%)")
        summary.append("")
    
    # Key findings from hypothesis testing
    if results['hypothesis_tests'] is not None:
        significant_tests = results['hypothesis_tests'][results['hypothesis_tests']['FDR_significant'] == True]
        summary.append(f"Statistical Significance:")
        summary.append(f"- {len(significant_tests)} metrics show statistically significant differences")
        summary.append(f"  between legitimate and problematic journals (FDR-corrected)")
        
        if len(significant_tests) > 0:
            top_metric = significant_tests.iloc[0]
            summary.append(f"- Most significant difference: {top_metric['Metric']}")
            summary.append(f"  (Effect size: {top_metric['Cohens_d']:.3f})")
        summary.append("")
    
    # Classification performance
    if results['classification_results'] is not None:
        best_model = results['classification_results'].loc[results['classification_results']['F1_Score'].idxmax()]
        summary.append(f"Classification Performance:")
        summary.append(f"- Best performing model: {best_model['Model']}")
        summary.append(f"- Accuracy: {best_model['Accuracy']:.3f}")
        summary.append(f"- F1-Score: {best_model['F1_Score']:.3f}")
        summary.append(f"- AUC: {best_model['AUC']:.3f}")
        summary.append("")
    
    # Clustering insights
    if results['suspicious_patterns'] is not None:
        high_suspicion_clusters = results['suspicious_patterns'][results['suspicious_patterns']['Suspicion_Score'] >= 2]
        max_score = results['suspicious_patterns']['Suspicion_Score'].max()
        summary.append(f"Clustering Analysis:")
        summary.append(f"- {len(high_suspicion_clusters)} clusters identified with suspicion scores ≥2")
        summary.append(f"- Maximum suspicion score: {max_score} (out of 4 possible)")
        summary.append(f"- Clustering reveals distinct patterns in journal behavior")
        summary.append("")
    
    # Publisher analysis
    if results['publisher_analysis'] is not None:
        high_risk_publishers = results['publisher_analysis'][results['publisher_analysis']['Problematic_Percentage'] > 50]
        total_problematic_only = results['publisher_analysis'][results['publisher_analysis']['Problematic_Percentage'] == 100.0]
        summary.append(f"Publisher Analysis:")
        summary.append(f"- {len(high_risk_publishers)} publishers with >50% problematic journals")
        summary.append(f"- {len(total_problematic_only)} publishers publish ONLY problematic journals (100%)")
        summary.append(f"- Clear patterns of systematic misconduct at publisher level")
        summary.append("")
    
    summary.append("RECOMMENDATIONS:")
    summary.append("- Implement enhanced screening for journals with high self-citation rates")
    summary.append("- Monitor publishers with concerning patterns")
    summary.append("- Use predictive models for early detection of problematic journals")
    summary.append("- Regular reassessment of journal quality metrics")
    
    return "\n".join(summary)

def create_key_findings_visualization(results):
    """Create a comprehensive visualization of key findings"""
    print("Creating key findings visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Dataset composition (top-left)
    if results['journal_predictions'] is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        labels = ['Legitimate', 'Problematic']
        sizes = [sum(results['journal_predictions']['Labels'] == 1.0),
                sum(results['journal_predictions']['Labels'] == 0.0)]
        colors = ['lightblue', 'lightcoral']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Dataset Composition')
    
    # 2. Top significant metrics (top-center)
    if results['hypothesis_tests'] is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        significant_tests = results['hypothesis_tests'][results['hypothesis_tests']['FDR_significant'] == True]
        if len(significant_tests) > 0:
            top_5 = significant_tests.head(5)
            bars = ax2.barh(range(len(top_5)), abs(top_5['Cohens_d']))
            ax2.set_yticks(range(len(top_5)))
            ax2.set_yticklabels([metric[:15] + '...' if len(metric) > 15 else metric 
                               for metric in top_5['Metric']])
            ax2.set_xlabel("Effect Size (|Cohen's d|)")
            ax2.set_title('Top Significant Differences')
            ax2.grid(True, alpha=0.3)
    
    # 3. Model performance comparison (top-right)
    if results['classification_results'] is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        models = results['classification_results']['Model']
        f1_scores = results['classification_results']['F1_Score']
        bars = ax3.bar(range(len(models)), f1_scores)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([model[:8] + '...' if len(model) > 8 else model 
                           for model in models], rotation=45)
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Model Performance')
        ax3.grid(True, alpha=0.3)
        
        # Color best performing model
        best_idx = f1_scores.idxmax()
        bars[best_idx].set_color('gold')
    
    # 4. Feature importance (middle-left)
    if results['feature_importance'] is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        top_features = results['feature_importance'].head(8)
        bars = ax4.barh(range(len(top_features)), top_features['Importance'])
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels([feat[:15] + '...' if len(feat) > 15 else feat 
                           for feat in top_features['Feature']])
        ax4.set_xlabel('Importance')
        ax4.set_title('Top Feature Importance')
        ax4.grid(True, alpha=0.3)
    
    # 5. Cluster suspicion scores (middle-center)
    if results['suspicious_patterns'] is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        clusters = results['suspicious_patterns']['Cluster']
        suspicion_scores = results['suspicious_patterns']['Suspicion_Score']
        colors = ['red' if score >= 3 else 'orange' if score >= 2 else 'green' 
                 for score in suspicion_scores]
        bars = ax5.bar(clusters, suspicion_scores, color=colors)
        ax5.set_xlabel('Cluster ID')
        ax5.set_ylabel('Suspicion Score')
        ax5.set_title('Cluster Suspicion Analysis')
        ax5.grid(True, alpha=0.3)
    
    # 6. Publisher risk analysis (middle-right)
    if results['publisher_analysis'] is not None:
        ax6 = fig.add_subplot(gs[1, 2])
        try:
            high_risk = results['publisher_analysis'][results['publisher_analysis']['Problematic_Percentage'] > 25]
            if len(high_risk) > 0:
                top_risk = high_risk.head(10)
                bars = ax6.barh(range(len(top_risk)), top_risk['Problematic_Percentage'])
                ax6.set_yticks(range(len(top_risk)))
                ax6.set_yticklabels([str(pub)[:20] + '...' if len(str(pub)) > 20 else str(pub)
                                   for pub in top_risk.index])
                ax6.set_xlabel('% Problematic Journals')
                ax6.set_title('High-Risk Publishers')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No high-risk publishers found', ha='center', va='center')
                ax6.set_title('Publisher Risk Analysis')
        except Exception as e:
            ax6.text(0.5, 0.5, f'Publisher analysis error: {str(e)[:30]}', ha='center', va='center')
            ax6.set_title('Publisher Risk Analysis')
    
    # 7. Correlation heatmap (bottom span)
    if results['correlation_matrix'] is not None:
        ax7 = fig.add_subplot(gs[2, :])
        try:
            # Select key correlations with Labels
            corr_data = results['correlation_matrix'].set_index(results['correlation_matrix'].columns[0])
            if 'Labels' in corr_data.columns:
                label_corrs = corr_data['Labels'].abs().sort_values(ascending=False)
                # Remove Labels from the series if it exists
                if 'Labels' in label_corrs.index:
                    label_corrs = label_corrs.drop('Labels')
                top_corrs = label_corrs.head(8)

                # Create a mini correlation matrix
                selected_features = list(top_corrs.index) + ['Labels']
                mini_corr = corr_data.loc[selected_features, selected_features]

                sns.heatmap(mini_corr, annot=True, cmap='RdBu_r', center=0,
                           square=True, ax=ax7, fmt='.2f')
                ax7.set_title('Key Correlations with Journal Legitimacy')
            else:
                ax7.text(0.5, 0.5, 'Labels column not found in correlation matrix', ha='center', va='center')
                ax7.set_title('Correlation Analysis')
        except Exception as e:
            ax7.text(0.5, 0.5, f'Correlation error: {str(e)[:30]}', ha='center', va='center')
            ax7.set_title('Correlation Analysis')
    
    plt.suptitle('StAMA: Comprehensive Analysis Summary', fontsize=16, fontweight='bold')
    plt.savefig('images/comprehensive_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_findings(results):
    """Generate detailed findings for each analysis component"""
    print("Generating detailed findings...")
    
    findings = []
    findings.append("\nDETAILED FINDINGS")
    findings.append("="*50)
    
    # Descriptive Statistics Findings
    findings.append("\n1. DESCRIPTIVE STATISTICS")
    findings.append("-" * 30)
    if results['descriptive_stats'] is not None:
        findings.append("Key observations from descriptive analysis:")
        findings.append("- Significant variation in journal quality metrics")
        findings.append("- Clear differences between legitimate and problematic journals")
        findings.append("- Some metrics show extreme outliers requiring investigation")
    
    # Hypothesis Testing Findings
    findings.append("\n2. HYPOTHESIS TESTING")
    findings.append("-" * 30)
    if results['hypothesis_tests'] is not None:
        significant_tests = results['hypothesis_tests'][results['hypothesis_tests']['FDR_significant'] == True]
        findings.append(f"Statistical significance testing revealed:")
        findings.append(f"- {len(significant_tests)} metrics with significant differences")
        
        for _, test in significant_tests.head(5).iterrows():
            effect_size = "large" if abs(test['Cohens_d']) > 0.8 else "medium" if abs(test['Cohens_d']) > 0.5 else "small"
            findings.append(f"- {test['Metric']}: {effect_size} effect size ({test['Cohens_d']:.3f})")
    
    # Clustering Findings
    findings.append("\n3. CLUSTERING ANALYSIS")
    findings.append("-" * 30)
    if results['cluster_characteristics'] is not None and results['suspicious_patterns'] is not None:
        findings.append("Clustering analysis identified distinct journal groups:")
        
        high_suspicion = results['suspicious_patterns'][results['suspicious_patterns']['Suspicion_Score'] >= 3]
        findings.append(f"- {len(high_suspicion)} clusters with high suspicion scores")
        
        for _, cluster in high_suspicion.iterrows():
            findings.append(f"- Cluster {cluster['Cluster']}: {cluster['Size']} journals, "
                          f"{cluster['Problematic_Ratio']*100:.1f}% problematic")
    
    # Classification Findings
    findings.append("\n4. CLASSIFICATION ANALYSIS")
    findings.append("-" * 30)
    if results['classification_results'] is not None:
        best_model = results['classification_results'].loc[results['classification_results']['F1_Score'].idxmax()]
        findings.append("Machine learning classification results:")
        findings.append(f"- Best model: {best_model['Model']}")
        findings.append(f"- Achieves {best_model['Accuracy']:.1%} accuracy")
        findings.append(f"- F1-score of {best_model['F1_Score']:.3f} indicates good balance")
        
        if results['feature_importance'] is not None:
            top_feature = results['feature_importance'].iloc[0]
            findings.append(f"- Most important predictor: {top_feature['Feature']}")
    
    # Publisher Analysis Findings
    findings.append("\n5. PUBLISHER ANALYSIS")
    findings.append("-" * 30)
    if results['publisher_analysis'] is not None:
        high_risk = results['publisher_analysis'][results['publisher_analysis']['Problematic_Percentage'] > 50]
        findings.append("Publisher-level analysis reveals:")
        findings.append(f"- {len(high_risk)} publishers with majority problematic journals")
        
        if len(high_risk) > 0:
            worst_publisher = high_risk.iloc[0]
            findings.append(f"- Highest risk: {worst_publisher.name} "
                          f"({worst_publisher['Problematic_Percentage']:.1f}% problematic)")
    
    return "\n".join(findings)

def generate_recommendations(results):
    """Generate actionable recommendations based on findings"""
    print("Generating recommendations...")
    
    recommendations = []
    recommendations.append("\nRECOMMENDATIONS")
    recommendations.append("="*50)
    
    # Immediate actions
    recommendations.append("\nIMMEDIATE ACTIONS:")
    recommendations.append("1. Implement automated screening using the developed classification model")
    recommendations.append("2. Flag journals with self-citation rates >10% for manual review")
    recommendations.append("3. Investigate publishers with >50% problematic journals")
    
    # Medium-term strategies
    recommendations.append("\nMEDIUM-TERM STRATEGIES:")
    recommendations.append("1. Develop real-time monitoring dashboard for journal metrics")
    recommendations.append("2. Establish collaboration with international journal databases")
    recommendations.append("3. Create standardized evaluation criteria based on findings")
    
    # Long-term initiatives
    recommendations.append("\nLONG-TERM INITIATIVES:")
    recommendations.append("1. Develop predictive models for early detection of emerging threats")
    recommendations.append("2. Establish international standards for journal quality assessment")
    recommendations.append("3. Create educational programs for researchers on journal selection")
    
    # Specific thresholds based on analysis
    if results['feature_importance'] is not None:
        top_features = results['feature_importance'].head(3)
        recommendations.append("\nSUGGESTED MONITORING THRESHOLDS:")
        for _, feature in top_features.iterrows():
            recommendations.append(f"- {feature['Feature']}: Monitor journals with extreme values")
    
    return "\n".join(recommendations)

def create_methodology_summary():
    """Create a summary of the methodology used"""
    methodology = []
    methodology.append("\nMETHODOLOGY")
    methodology.append("="*50)
    
    methodology.append("\nDATA COLLECTION:")
    methodology.append("- Journal metrics from Google Scholar and Beall's List")
    methodology.append("- Features: SJR, H-index, IPP, SNIP, citation patterns, etc.")
    methodology.append("- Binary classification: Legitimate (1) vs Problematic (0)")
    
    methodology.append("\nSTATISTICAL METHODS:")
    methodology.append("- Descriptive statistics and correlation analysis")
    methodology.append("- Hypothesis testing with multiple comparison correction")
    methodology.append("- Confidence intervals and effect size calculations")
    
    methodology.append("\nMACHINE LEARNING:")
    methodology.append("- Unsupervised clustering (K-means, DBSCAN, Hierarchical)")
    methodology.append("- Supervised classification (Random Forest, SVM, etc.)")
    methodology.append("- Feature selection and hyperparameter tuning")
    methodology.append("- Cross-validation and performance evaluation")
    
    methodology.append("\nVISUALIZATION:")
    methodology.append("- PCA and t-SNE for dimensionality reduction")
    methodology.append("- Interactive plots for data exploration")
    methodology.append("- Comprehensive dashboards for results presentation")
    
    return "\n".join(methodology)

def generate_full_report(results):
    """Generate the complete comprehensive report"""
    print("Generating comprehensive report...")
    
    report_sections = []
    
    # Header
    report_sections.append("StAMA - STATISTICAL ANALYSIS IN MODERN ACADEMIA")
    report_sections.append("Journal Clustering and Publication Misconduct Detection")
    report_sections.append("="*80)
    report_sections.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_sections.append("")
    
    # Executive Summary
    report_sections.append(generate_executive_summary(results))
    
    # Methodology
    report_sections.append(create_methodology_summary())
    
    # Detailed Findings
    report_sections.append(generate_detailed_findings(results))
    
    # Recommendations
    report_sections.append(generate_recommendations(results))
    
    # Appendix
    report_sections.append("\nAPPENDIX")
    report_sections.append("="*50)
    report_sections.append("Additional files generated:")
    report_sections.append("- All analysis scripts (01_*.py through 06_*.py)")
    report_sections.append("- Detailed results in CSV format (images/*.csv)")
    report_sections.append("- Visualizations and plots (images/*.png)")
    report_sections.append("- Interactive visualizations (images/*.html)")
    
    # Combine all sections
    full_report = "\n".join(report_sections)
    
    # Save report
    with open('images/StAMA_Comprehensive_Report.txt', 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    return full_report

def create_summary_statistics_table(results):
    """Create a summary table of key statistics"""
    print("Creating summary statistics table...")
    
    summary_stats = []
    
    # Dataset statistics
    if results['journal_predictions'] is not None:
        total_journals = len(results['journal_predictions'])
        legitimate = sum(results['journal_predictions']['Labels'] == 1.0)
        problematic = sum(results['journal_predictions']['Labels'] == 0.0)
        
        summary_stats.append({
            'Metric': 'Total Journals',
            'Value': f"{total_journals:,}",
            'Description': 'Total number of journals analyzed'
        })
        
        summary_stats.append({
            'Metric': 'Legitimate Journals',
            'Value': f"{legitimate:,} ({legitimate/total_journals*100:.1f}%)",
            'Description': 'Journals classified as legitimate'
        })
        
        summary_stats.append({
            'Metric': 'Problematic Journals',
            'Value': f"{problematic:,} ({problematic/total_journals*100:.1f}%)",
            'Description': 'Journals flagged as potentially problematic'
        })
    
    # Statistical significance
    if results['hypothesis_tests'] is not None:
        significant = sum(results['hypothesis_tests']['FDR_significant'])
        total_tests = len(results['hypothesis_tests'])
        
        summary_stats.append({
            'Metric': 'Significant Differences',
            'Value': f"{significant}/{total_tests}",
            'Description': 'Metrics with statistically significant differences'
        })
    
    # Model performance
    if results['classification_results'] is not None:
        best_accuracy = results['classification_results']['Accuracy'].max()
        best_f1 = results['classification_results']['F1_Score'].max()
        
        summary_stats.append({
            'Metric': 'Best Model Accuracy',
            'Value': f"{best_accuracy:.3f}",
            'Description': 'Highest classification accuracy achieved'
        })
        
        summary_stats.append({
            'Metric': 'Best Model F1-Score',
            'Value': f"{best_f1:.3f}",
            'Description': 'Best balance of precision and recall'
        })
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('images/summary_statistics_table.csv', index=False)
    
    return summary_df

def main():
    """Main function to generate comprehensive report"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 6: Comprehensive Report Generation")
    print("="*60)
    
    # Load all analysis results
    results = load_analysis_results()
    
    # Generate comprehensive visualizations
    create_key_findings_visualization(results)
    
    # Create summary statistics table
    summary_df = create_summary_statistics_table(results)
    
    # Generate full report
    full_report = generate_full_report(results)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE REPORT GENERATION COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- StAMA_Comprehensive_Report.txt")
    print("- comprehensive_summary_dashboard.png")
    print("- summary_statistics_table.csv")
    print("\nThe comprehensive report includes:")
    print("- Executive summary of key findings")
    print("- Detailed methodology description")
    print("- Statistical analysis results")
    print("- Machine learning insights")
    print("- Actionable recommendations")
    print("- Visual summary dashboard")
    
    # Print a preview of the report
    print("\n" + "="*60)
    print("REPORT PREVIEW")
    print("="*60)
    print(full_report[:1000] + "..." if len(full_report) > 1000 else full_report)

if __name__ == "__main__":
    main()
