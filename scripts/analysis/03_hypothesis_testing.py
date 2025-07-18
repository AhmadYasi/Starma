"""
StAMA - Statistical Analysis in Modern Academia
Script 3: Hypothesis Testing and Confidence Intervals

This script performs comprehensive hypothesis testing to identify statistically
significant differences between legitimate and potentially problematic journals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the journal data"""
    print("Loading journal data for hypothesis testing...")
    df = pd.read_csv('../../data/cleaned/Data_with_Labels.csv')
    return df

def test_normality(data, name):
    """Test normality of data using multiple tests"""
    results = {}
    
    # Shapiro-Wilk test (for smaller samples)
    if len(data) <= 5000:
        stat, p_value = stats.shapiro(data)
        results['Shapiro-Wilk'] = {'statistic': stat, 'p_value': p_value}
    
    # Kolmogorov-Smirnov test
    stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    results['Kolmogorov-Smirnov'] = {'statistic': stat, 'p_value': p_value}
    
    # Anderson-Darling test
    result = stats.anderson(data, dist='norm')
    results['Anderson-Darling'] = {'statistic': result.statistic, 'critical_values': result.critical_values}
    
    return results

def perform_two_sample_tests(df):
    """Perform two-sample tests comparing legitimate vs problematic journals"""
    print("\n" + "="*60)
    print("TWO-SAMPLE HYPOTHESIS TESTS")
    print("="*60)
    
    # Key metrics to test
    test_metrics = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                   'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)', 'Ref. / Doc.',
                   'Total Docs. (2023)', 'Total Cites (3years)', 'Coverage_Duration']
    
    results = []
    
    for metric in test_metrics:
        print(f"\nTesting: {metric}")
        print("-" * 40)
        
        # Separate data by label
        legitimate = df[df['Labels'] == 1.0][metric].dropna()
        problematic = df[df['Labels'] == 0.0][metric].dropna()
        
        if len(legitimate) == 0 or len(problematic) == 0:
            print(f"Insufficient data for {metric}")
            continue
        
        # Basic statistics
        leg_mean, leg_std = legitimate.mean(), legitimate.std()
        prob_mean, prob_std = problematic.mean(), problematic.std()
        
        print(f"Legitimate journals: n={len(legitimate)}, mean={leg_mean:.4f}, std={leg_std:.4f}")
        print(f"Problematic journals: n={len(problematic)}, mean={prob_mean:.4f}, std={prob_std:.4f}")
        
        # Test normality
        leg_normality = test_normality(legitimate, f"{metric}_legitimate")
        prob_normality = test_normality(problematic, f"{metric}_problematic")
        
        # Check if data is approximately normal (using Shapiro-Wilk p-value > 0.05)
        leg_normal = leg_normality.get('Shapiro-Wilk', {}).get('p_value', 0) > 0.05 if len(legitimate) <= 5000 else False
        prob_normal = prob_normality.get('Shapiro-Wilk', {}).get('p_value', 0) > 0.05 if len(problematic) <= 5000 else False
        
        # Perform appropriate tests
        test_results = {}
        
        # 1. Mann-Whitney U test (non-parametric)
        u_stat, u_p = mannwhitneyu(legitimate, problematic, alternative='two-sided')
        test_results['Mann-Whitney U'] = {'statistic': u_stat, 'p_value': u_p}
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p={u_p:.6f}")
        
        # 2. Welch's t-test (doesn't assume equal variances)
        t_stat, t_p = stats.ttest_ind(legitimate, problematic, equal_var=False)
        test_results['Welch t-test'] = {'statistic': t_stat, 'p_value': t_p}
        print(f"Welch's t-test: t={t_stat:.4f}, p={t_p:.6f}")
        
        # 3. Levene's test for equal variances
        lev_stat, lev_p = stats.levene(legitimate, problematic)
        test_results['Levene (equal var)'] = {'statistic': lev_stat, 'p_value': lev_p}
        print(f"Levene's test (equal variances): F={lev_stat:.4f}, p={lev_p:.6f}")
        
        # 4. Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(legitimate) - 1) * leg_std**2 + (len(problematic) - 1) * prob_std**2) / 
                           (len(legitimate) + len(problematic) - 2))
        cohens_d = (leg_mean - prob_mean) / pooled_std
        test_results['Cohen_d'] = cohens_d
        print(f"Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d) < 0.5:
            effect_size = "Medium"
        elif abs(cohens_d) < 0.8:
            effect_size = "Large"
        else:
            effect_size = "Very Large"
        
        print(f"Effect size interpretation: {effect_size}")
        
        # Store results
        result_row = {
            'Metric': metric,
            'Legitimate_n': len(legitimate),
            'Legitimate_mean': leg_mean,
            'Legitimate_std': leg_std,
            'Problematic_n': len(problematic),
            'Problematic_mean': prob_mean,
            'Problematic_std': prob_std,
            'Mann_Whitney_U': u_stat,
            'Mann_Whitney_p': u_p,
            'Welch_t': t_stat,
            'Welch_t_p': t_p,
            'Levene_F': lev_stat,
            'Levene_p': lev_p,
            'Cohens_d': cohens_d,
            'Effect_size': effect_size
        }
        
        results.append(result_row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('images/hypothesis_test_results.csv', index=False)
    
    return results_df

def confidence_intervals_analysis(df):
    """Calculate confidence intervals for key metrics"""
    print("\n" + "="*60)
    print("CONFIDENCE INTERVALS ANALYSIS")
    print("="*60)
    
    key_metrics = ['SJR', 'H index', 'Self-Cites/Total Cites (3years)', 
                   'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)']
    
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    ci_results = []
    
    for metric in key_metrics:
        print(f"\nConfidence Intervals for {metric}:")
        print("-" * 40)
        
        for label, label_name in [(1.0, 'Legitimate'), (0.0, 'Problematic')]:
            data = df[df['Labels'] == label][metric].dropna()
            
            if len(data) == 0:
                continue
            
            n = len(data)
            mean = data.mean()
            std = data.std()
            se = std / np.sqrt(n)
            
            # t-distribution critical value
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            
            # Confidence interval
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se
            
            print(f"{label_name}: n={n}, mean={mean:.4f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Bootstrap confidence interval
            bootstrap_means = []
            n_bootstrap = 1000
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=n, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_ci_lower = np.percentile(bootstrap_means, 2.5)
            bootstrap_ci_upper = np.percentile(bootstrap_means, 97.5)
            
            print(f"Bootstrap 95% CI: [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")
            
            ci_results.append({
                'Metric': metric,
                'Journal_Type': label_name,
                'n': n,
                'Mean': mean,
                'Std': std,
                'SE': se,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Bootstrap_CI_Lower': bootstrap_ci_lower,
                'Bootstrap_CI_Upper': bootstrap_ci_upper
            })
    
    # Save confidence intervals
    ci_df = pd.DataFrame(ci_results)
    ci_df.to_csv('images/confidence_intervals.csv', index=False)
    
    return ci_df

def correlation_hypothesis_tests(df):
    """Test correlations between variables and journal legitimacy"""
    print("\n" + "="*60)
    print("CORRELATION HYPOTHESIS TESTS")
    print("="*60)
    
    numerical_cols = ['SJR', 'H index', 'IPP', 'SNIP', 'Self-Cites/Total Cites (3years)',
                     'Uncited Docs./Total Docs. (3years)', 'Cites / Doc. (2years)', 'Ref. / Doc.',
                     'Total Docs. (2023)', 'Total Cites (3years)', 'Coverage_Duration']
    
    correlation_results = []
    
    for col in numerical_cols:
        data_col = df[col].dropna()
        labels_col = df.loc[data_col.index, 'Labels']
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(data_col, labels_col)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(data_col, labels_col)
        
        print(f"\n{col}:")
        print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
        print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.6f}")
        
        correlation_results.append({
            'Variable': col,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_rho': spearman_r,
            'Spearman_p': spearman_p
        })
    
    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv('images/correlation_hypothesis_tests.csv', index=False)
    
    return corr_df

def multiple_testing_correction(results_df):
    """Apply multiple testing corrections"""
    print("\n" + "="*60)
    print("MULTIPLE TESTING CORRECTION")
    print("="*60)
    
    # Extract p-values for correction
    p_values = results_df['Mann_Whitney_p'].values
    
    # Bonferroni correction
    bonferroni_alpha = 0.05 / len(p_values)
    bonferroni_significant = p_values < bonferroni_alpha
    
    # Benjamini-Hochberg (FDR) correction
    from statsmodels.stats.multitest import multipletests
    fdr_reject, fdr_pvals, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    # Add corrections to results
    results_df['Bonferroni_alpha'] = bonferroni_alpha
    results_df['Bonferroni_significant'] = bonferroni_significant
    results_df['FDR_corrected_p'] = fdr_pvals
    results_df['FDR_significant'] = fdr_reject
    
    print(f"Original alpha level: 0.05")
    print(f"Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
    print(f"Number of tests: {len(p_values)}")
    print(f"Significant after Bonferroni: {sum(bonferroni_significant)}")
    print(f"Significant after FDR: {sum(fdr_reject)}")
    
    # Save corrected results
    results_df.to_csv('images/hypothesis_test_results_corrected.csv', index=False)
    
    return results_df

def create_hypothesis_visualizations(results_df, ci_df):
    """Create visualizations for hypothesis test results"""
    print("\nCreating hypothesis test visualizations...")
    
    # 1. Effect sizes plot
    plt.figure(figsize=(12, 8))
    metrics = results_df['Metric']
    effect_sizes = results_df['Cohens_d']
    colors = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green' 
              for d in effect_sizes]
    
    bars = plt.barh(range(len(metrics)), effect_sizes, color=colors)
    plt.yticks(range(len(metrics)), metrics)
    plt.xlabel("Cohen's d (Effect Size)")
    plt.title("Effect Sizes: Legitimate vs Problematic Journals")
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    plt.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/effect_sizes_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. P-values plot
    plt.figure(figsize=(12, 8))
    p_values = results_df['Mann_Whitney_p']
    significant = p_values < 0.05
    colors = ['red' if sig else 'blue' for sig in significant]
    
    bars = plt.barh(range(len(metrics)), -np.log10(p_values), color=colors)
    plt.yticks(range(len(metrics)), metrics)
    plt.xlabel("-log10(p-value)")
    plt.title("Statistical Significance: Mann-Whitney U Tests")
    plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')
    plt.axvline(x=-np.log10(0.01), color='orange', linestyle='--', label='α = 0.01')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/p_values_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all hypothesis testing analyses"""
    # Create images directory
    import os
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 3: Hypothesis Testing and Confidence Intervals")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Perform analyses
    results_df = perform_two_sample_tests(df)
    ci_df = confidence_intervals_analysis(df)
    corr_df = correlation_hypothesis_tests(df)
    corrected_results_df = multiple_testing_correction(results_df)
    create_hypothesis_visualizations(corrected_results_df, ci_df)
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved in 'images' directory:")
    print("- hypothesis_test_results.csv")
    print("- hypothesis_test_results_corrected.csv")
    print("- confidence_intervals.csv")
    print("- correlation_hypothesis_tests.csv")
    print("- effect_sizes_plot.png")
    print("- p_values_plot.png")

if __name__ == "__main__":
    main()
