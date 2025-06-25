# StAMA - Statistical Analysis in Modern Academia

## Journal Clustering and Publication Misconduct Detection

This project implements comprehensive statistical methods to analyze journal data and identify patterns that may indicate unethical practices in academic publishing. The analysis combines descriptive statistics, hypothesis testing, clustering, and machine learning classification to provide quantitative insights into publication misconduct.

## 🎯 Objectives

- **Descriptive Statistics**: Analyze journal metrics and identify patterns
- **Data Visualization**: Create comprehensive visualizations of journal characteristics
- **Hypothesis Testing**: Test for significant differences between legitimate and problematic journals
- **Clustering Analysis**: Identify groups of journals with similar characteristics
- **Classification**: Develop predictive models to detect potentially problematic journals
- **Impact Assessment**: Examine the implications for the scientific community

## 📊 Data Sources

- **Google Scholar**: Journal metrics including SJR, H-index, IPP, SNIP
- **Beall's List**: Classification of potentially predatory journals
- **Citation Data**: Self-citation rates, uncited documents, references per document

## 🔧 Installation

1. **Clone or download the project files**
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure your data file is named `Data_with_Labels.csv` and placed in the project directory**

## 🚀 Usage

### Option 1: Run All Analyses (Recommended)
```bash
python run_all_analyses.py
```

This will execute all analysis scripts in sequence and generate a comprehensive report.

### Option 2: Run Individual Scripts
```bash
python 01_descriptive_statistics.py
python 02_data_visualization.py
python 03_hypothesis_testing.py
python 04_clustering_analysis.py
python 05_classification_analysis.py
python 06_comprehensive_report.py
```

## 📁 Project Structure

```
StAMA/
├── Data_with_Labels.csv          # Input data file
├── run_all_analyses.py           # Master script to run all analyses
├── 01_descriptive_statistics.py  # Basic statistical analysis
├── 02_data_visualization.py      # Data visualization
├── 03_hypothesis_testing.py      # Hypothesis testing & confidence intervals
├── 04_clustering_analysis.py     # Clustering analysis
├── 05_classification_analysis.py # Classification models
├── 06_comprehensive_report.py    # Report generation
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── images/                       # Output directory (created automatically)
    ├── *.csv                     # Statistical results
    ├── *.png                     # Visualizations
    ├── *.html                    # Interactive plots
    └── StAMA_Comprehensive_Report.txt
```

## 📈 Analysis Components

### 1. Descriptive Statistics (`01_descriptive_statistics.py`)
- **Basic descriptive statistics** for all journal metrics
- **Missing data analysis** and handling strategies
- **Distribution analysis** with normality testing
- **Correlation analysis** between variables
- **Outlier detection** using IQR method

**Outputs**: 
- `descriptive_statistics_*.csv`
- `correlation_matrix.png/.csv`
- `distribution_analysis.png`
- `outlier_analysis.csv`

### 2. Data Visualization (`02_data_visualization.py`)
- **Comparison plots** between legitimate and problematic journals
- **Distribution comparisons** for key metrics
- **Scatter plot analysis** to identify patterns
- **Publisher analysis** showing risk levels
- **Citation pattern analysis** for misconduct detection
- **Interactive visualizations** using Plotly

**Outputs**:
- `journal_comparison_boxplots.png`
- `distribution_comparisons.png`
- `scatter_plot_analysis.png`
- `publisher_analysis.png/.csv`
- `citation_patterns_analysis.png`
- `interactive_*.html`

### 3. Hypothesis Testing (`03_hypothesis_testing.py`)
- **Two-sample tests** (Mann-Whitney U, Welch's t-test)
- **Effect size calculations** (Cohen's d)
- **Confidence intervals** (parametric and bootstrap)
- **Correlation hypothesis tests** (Pearson and Spearman)
- **Multiple testing correction** (Bonferroni and FDR)

**Outputs**:
- `hypothesis_test_results_corrected.csv`
- `confidence_intervals.csv`
- `correlation_hypothesis_tests.csv`
- `effect_sizes_plot.png`
- `p_values_plot.png`

### 4. Clustering Analysis (`04_clustering_analysis.py`)
- **Optimal cluster determination** using multiple methods
- **K-means clustering** with hyperparameter optimization
- **DBSCAN clustering** for density-based grouping
- **Hierarchical clustering** with dendrogram analysis
- **PCA and t-SNE visualizations** for dimensionality reduction
- **Suspicious pattern identification** in clusters

**Outputs**:
- `optimal_clusters_analysis.png`
- `*_pca_visualization.png`
- `*_tsne_visualization.png`
- `cluster_characteristics.csv/.png`
- `suspicious_patterns_analysis.csv`

### 5. Classification Analysis (`05_classification_analysis.py`)
- **Feature selection** using statistical tests
- **Multiple classification models** (Random Forest, SVM, etc.)
- **Hyperparameter tuning** with grid search
- **Model evaluation** with cross-validation
- **ROC curves and confusion matrices**
- **Feature importance analysis**
- **Journal prediction** with probability scores

**Outputs**:
- `classification_results.csv`
- `feature_importance_*.png/.csv`
- `model_comparison_plots.png`
- `roc_curves_comparison.png`
- `confusion_matrices.png`
- `journal_predictions.csv`

### 6. Comprehensive Report (`06_comprehensive_report.py`)
- **Executive summary** of key findings
- **Detailed methodology** description
- **Statistical results** compilation
- **Actionable recommendations**
- **Visual summary dashboard**

**Outputs**:
- `StAMA_Comprehensive_Report.txt`
- `comprehensive_summary_dashboard.png`
- `summary_statistics_table.csv`

## 🔍 Key Features

### Statistical Rigor
- Multiple hypothesis testing with correction for false discovery rate
- Effect size calculations for practical significance
- Bootstrap confidence intervals for robust estimation
- Cross-validation for model reliability

### Machine Learning
- Unsupervised clustering to identify natural groupings
- Supervised classification for prediction
- Feature selection and importance analysis
- Hyperparameter optimization

### Visualization
- Static plots using matplotlib/seaborn
- Interactive visualizations using Plotly
- Dimensionality reduction visualizations
- Comprehensive dashboards

## 📊 Expected Results

The analysis will identify:
- **Statistical differences** between legitimate and problematic journals
- **Clustering patterns** that reveal journal groupings
- **Predictive models** for detecting misconduct
- **Publisher-level risk assessment**
- **Key metrics** most indicative of journal quality
- **Actionable recommendations** for the scientific community

## 🎯 Impact on Scientific Community

This analysis provides:
- **Quantitative methods** for journal evaluation
- **Evidence-based tools** for detecting publication misconduct
- **Data-driven insights** into citation manipulation
- **Protective measures** for research integrity
- **Standards** for journal quality assessment

## 🔧 Technical Requirements

- Python 3.7+
- Required packages (see `requirements.txt`)
- Minimum 4GB RAM (8GB recommended for large datasets)
- ~500MB disk space for outputs

## 📝 Data Format

The input CSV file should contain the following columns:
- `Title`: Journal title
- `SJR`: SCImago Journal Rank
- `H index`: Hirsch index
- `IPP`: Impact per Publication
- `SNIP`: Source Normalized Impact per Paper
- `Self-Cites/Total Cites (3years)`: Self-citation ratio
- `Uncited Docs./Total Docs. (3years)`: Uncited documents ratio
- `Cites / Doc. (2years)`: Citations per document
- `Ref. / Doc.`: References per document
- `Publisher`: Publisher name
- `Labels`: 1.0 for legitimate, 0.0 for problematic journals


**StAMA - Protecting Scientific Integrity Through Statistical Analysis**
