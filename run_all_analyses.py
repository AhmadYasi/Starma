"""
StAMA - Statistical Analysis in Modern Academia
Master Script: Run All Analyses

This script runs all the statistical analyses in sequence and generates
a comprehensive report on journal clustering and publication misconduct detection.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header():
    """Print the project header"""
    print("="*80)
    print("StAMA - STATISTICAL ANALYSIS IN MODERN ACADEMIA")
    print("Journal Clustering and Publication Misconduct Detection")
    print("="*80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def print_section_header(section_num, title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"STEP {section_num}: {title}")
    print(f"{'='*60}")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\nRunning {script_name}...")
    print(f"Description: {description}")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {script_name} completed successfully in {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    except Exception as e:
        print(f"✗ Unexpected error running {script_name}: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
        'sklearn', 'plotly', 'statsmodels'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - NOT FOUND")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All dependencies are available!")
    return True

def check_data_file():
    """Check if the data file exists"""
    data_file = "Data_with_Labels.csv"
    if os.path.exists(data_file):
        print(f"✓ Data file found: {data_file}")
        return True
    else:
        print(f"✗ Data file not found: {data_file}")
        print("Please ensure the CSV file is in the current directory.")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("Creating directory structure...")
    
    directories = ['images']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def main():
    """Main function to run all analyses"""
    print_header()
    
    # Check prerequisites
    print_section_header(0, "CHECKING PREREQUISITES")
    
    if not check_dependencies():
        print("Please install missing dependencies before running the analysis.")
        return False
    
    if not check_data_file():
        print("Please ensure the data file is available before running the analysis.")
        return False
    
    create_directory_structure()
    
    # Define analysis scripts
    analysis_scripts = [
        {
            'script': '01_descriptive_statistics.py',
            'title': 'DESCRIPTIVE STATISTICS ANALYSIS',
            'description': 'Basic statistical analysis, distributions, correlations, and outlier detection'
        },
        {
            'script': '02_data_visualization.py',
            'title': 'DATA VISUALIZATION',
            'description': 'Comprehensive visualizations comparing legitimate vs problematic journals'
        },
        {
            'script': '03_hypothesis_testing.py',
            'title': 'HYPOTHESIS TESTING & CONFIDENCE INTERVALS',
            'description': 'Statistical significance testing and confidence interval analysis'
        },
        {
            'script': '04_clustering_analysis.py',
            'title': 'CLUSTERING ANALYSIS',
            'description': 'Unsupervised learning to identify journal clusters and suspicious patterns'
        },
        {
            'script': '05_classification_analysis.py',
            'title': 'CLASSIFICATION ANALYSIS',
            'description': 'Supervised learning to predict journal legitimacy and identify key features'
        },
        {
            'script': '06_comprehensive_report.py',
            'title': 'COMPREHENSIVE REPORT GENERATION',
            'description': 'Generate final report summarizing all findings and recommendations'
        }
    ]
    
    # Track success/failure
    successful_scripts = []
    failed_scripts = []
    
    # Run each analysis script
    for i, script_info in enumerate(analysis_scripts, 1):
        print_section_header(i, script_info['title'])
        
        success = run_script(script_info['script'], script_info['description'])
        
        if success:
            successful_scripts.append(script_info['script'])
        else:
            failed_scripts.append(script_info['script'])
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful scripts: {len(successful_scripts)}/{len(analysis_scripts)}")
    
    if successful_scripts:
        print("\n✓ Successfully completed:")
        for script in successful_scripts:
            print(f"  - {script}")
    
    if failed_scripts:
        print("\n✗ Failed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
        print("\nPlease check the error messages above and resolve any issues.")
    
    # Results summary
    if len(successful_scripts) == len(analysis_scripts):
        print("\n🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("📁 images/ directory contains:")
        print("  - Statistical analysis results (CSV files)")
        print("  - Visualizations and plots (PNG files)")
        print("  - Interactive visualizations (HTML files)")
        print("  - Comprehensive report (TXT file)")
        
        print("\n📊 Key deliverables:")
        print("  - Descriptive statistics and correlations")
        print("  - Hypothesis testing results with effect sizes")
        print("  - Cluster analysis identifying suspicious patterns")
        print("  - Classification models for predicting journal legitimacy")
        print("  - Publisher risk analysis")
        print("  - Comprehensive summary report")
        
        print("\n📈 Impact on scientific community:")
        print("  - Quantitative methods for detecting publication misconduct")
        print("  - Evidence-based approach to journal evaluation")
        print("  - Tools for protecting research integrity")
        print("  - Data-driven insights into citation manipulation")
        
    else:
        print(f"\n⚠️  {len(failed_scripts)} script(s) failed. Please resolve issues and re-run.")
    
    print("\n" + "="*80)
    
    return len(failed_scripts) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
