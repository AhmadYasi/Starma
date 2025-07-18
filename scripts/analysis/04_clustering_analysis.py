"""
StAMA - Statistical Analysis in Modern Academia
Script 4: Clustering Analysis

This script performs clustering analysis to identify patterns and groups in journal data
that may indicate unethical practices or citation manipulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare data for clustering analysis"""
    print("Loading and preparing data for clustering analysis...")
    df = pd.read_csv('../../data/cleaned/Data_with_Labels.csv')
    
    # Select features for clustering
    clustering_features = [
        'SJR', 'H index', 'IPP', 'SNIP',
        'Self-Cites/Total Cites (3years)',
        'Uncited Docs./Total Docs. (3years)',
        'Cites / Doc. (2years)',
        'Ref. / Doc.',
        'Total Docs. (2023)',
        'Total Cites (3years)',
        'Coverage_Duration'
    ]
    
    # Create feature matrix
    X = df[clustering_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X = X[(X[col] >= mean - 3*std) & (X[col] <= mean + 3*std)]
    
    # Get corresponding labels and titles
    y_true = df.loc[X.index, 'Labels'].values
    titles = df.loc[X.index, 'Title'].values
    publishers = df.loc[X.index, 'Publisher'].values
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Features used: {clustering_features}")
    
    return X, y_true, titles, publishers, clustering_features

def scale_features(X):
    """Scale features using different methods"""
    print("Scaling features...")
    
    # Standard scaling
    scaler_standard = StandardScaler()
    X_standard = scaler_standard.fit_transform(X)
    
    # Robust scaling (less sensitive to outliers)
    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X)
    
    return X_standard, X_robust, scaler_standard, scaler_robust

def determine_optimal_clusters(X_scaled, max_clusters=15):
    """Determine optimal number of clusters using multiple methods"""
    print("Determining optimal number of clusters...")
    
    # Elbow method
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Elbow plot
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score plot
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score vs Number of Clusters')
    axes[1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz score plot
    axes[2].plot(K_range, calinski_scores, 'go-')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Score vs Number of Clusters')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'k': K_range,
        'inertia': inertias,
        'silhouette_score': silhouette_scores,
        'calinski_harabasz_score': calinski_scores
    })
    results_df.to_csv('images/cluster_optimization_results.csv', index=False)
    
    return optimal_k, results_df

def perform_kmeans_clustering(X_scaled, optimal_k, y_true):
    """Perform K-means clustering"""
    print(f"Performing K-means clustering with k={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    
    print(f"K-means Results:")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    
    return kmeans, cluster_labels, silhouette_avg

def perform_dbscan_clustering(X_scaled, y_true):
    """Perform DBSCAN clustering"""
    print("Performing DBSCAN clustering...")
    
    # Try different eps values
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    best_eps = None
    best_score = -1
    best_labels = None
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        # Skip if all points are noise or all in one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        
        # Calculate silhouette score (excluding noise points)
        if len(set(labels)) > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_labels = labels
    
    if best_labels is not None:
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = list(best_labels).count(-1)
        ari_score = adjusted_rand_score(y_true, best_labels)
        
        print(f"DBSCAN Results (eps={best_eps}):")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Silhouette Score: {best_score:.4f}")
        print(f"Adjusted Rand Index: {ari_score:.4f}")
    else:
        print("DBSCAN: No suitable clustering found")
        best_labels = np.zeros(len(X_scaled))
    
    return best_labels, best_eps

def perform_hierarchical_clustering(X_scaled, optimal_k, y_true):
    """Perform hierarchical clustering"""
    print(f"Performing hierarchical clustering with k={optimal_k}...")
    
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    cluster_labels = hierarchical.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    
    print(f"Hierarchical Clustering Results:")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    
    return hierarchical, cluster_labels

def visualize_clusters_pca(X_scaled, cluster_labels, y_true, method_name):
    """Visualize clusters using PCA"""
    print(f"Creating PCA visualization for {method_name}...")
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot clusters
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title(f'{method_name} Clusters (PCA)')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot true labels
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='RdYlBu', alpha=0.7)
    axes[1].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1].set_title('True Labels (PCA)')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f'images/{method_name.lower().replace(" ", "_")}_pca_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca, X_pca

def visualize_clusters_tsne(X_scaled, cluster_labels, y_true, method_name):
    """Visualize clusters using t-SNE"""
    print(f"Creating t-SNE visualization for {method_name}...")
    
    # Perform t-SNE (use a subset if data is too large)
    if len(X_scaled) > 1000:
        indices = np.random.choice(len(X_scaled), 1000, replace=False)
        X_subset = X_scaled[indices]
        cluster_subset = cluster_labels[indices]
        y_subset = y_true[indices]
    else:
        X_subset = X_scaled
        cluster_subset = cluster_labels
        y_subset = y_true
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_subset)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot clusters
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_subset, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].set_title(f'{method_name} Clusters (t-SNE)')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot true labels
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='RdYlBu', alpha=0.7)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('True Labels (t-SNE)')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f'images/{method_name.lower().replace(" ", "_")}_tsne_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cluster_characteristics(X, cluster_labels, feature_names, titles, publishers):
    """Analyze characteristics of each cluster"""
    print("Analyzing cluster characteristics...")
    
    # Create DataFrame with clusters
    df_clusters = X.copy()
    df_clusters['Cluster'] = cluster_labels
    df_clusters['Title'] = titles
    df_clusters['Publisher'] = publishers
    
    # Calculate cluster statistics
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
        
        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
        
        stats = {
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Percentage': len(cluster_data) / len(df_clusters) * 100
        }
        
        # Add mean values for each feature
        for feature in feature_names:
            stats[f'{feature}_mean'] = cluster_data[feature].mean()
            stats[f'{feature}_std'] = cluster_data[feature].std()
        
        cluster_stats.append(stats)
    
    # Convert to DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_csv('images/cluster_characteristics.csv', index=False)
    
    # Create heatmap of cluster means
    feature_means = cluster_stats_df[[col for col in cluster_stats_df.columns if col.endswith('_mean')]]
    feature_means.columns = [col.replace('_mean', '') for col in feature_means.columns]
    feature_means.index = cluster_stats_df['Cluster']
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_means.T, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title('Cluster Characteristics Heatmap (Mean Values)')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('images/cluster_characteristics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_stats_df, df_clusters

def identify_suspicious_patterns(df_clusters, y_true):
    """Identify potentially suspicious patterns in clusters"""
    print("Identifying suspicious patterns...")

    suspicious_patterns = []

    for cluster_id in sorted(set(df_clusters['Cluster'])):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]

        # Use only indices that exist in y_true
        valid_indices = [i for i in cluster_data.index if i < len(y_true)]
        if len(valid_indices) == 0:
            continue

        # Calculate proportion of problematic journals in cluster
        true_labels_cluster = y_true[valid_indices]
        problematic_ratio = np.mean(true_labels_cluster == 0.0)
        
        # Identify suspicious characteristics
        high_self_citation = cluster_data['Self-Cites/Total Cites (3years)'].mean() > 0.1
        high_uncited = cluster_data['Uncited Docs./Total Docs. (3years)'].mean() > 0.5
        low_sjr = cluster_data['SJR'].mean() < 1.0
        
        pattern = {
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Problematic_Ratio': problematic_ratio,
            'High_Self_Citation': high_self_citation,
            'High_Uncited_Docs': high_uncited,
            'Low_SJR': low_sjr,
            'Suspicion_Score': sum([high_self_citation, high_uncited, low_sjr, problematic_ratio > 0.5])
        }
        
        suspicious_patterns.append(pattern)
    
    # Convert to DataFrame and sort by suspicion score
    suspicious_df = pd.DataFrame(suspicious_patterns)
    suspicious_df = suspicious_df.sort_values('Suspicion_Score', ascending=False)
    
    print("\nSuspicious Pattern Analysis:")
    print(suspicious_df)
    
    suspicious_df.to_csv('images/suspicious_patterns_analysis.csv', index=False)
    
    return suspicious_df

def main():
    """Main function to run clustering analysis"""
    # Create images directory
    import os
    os.makedirs('images', exist_ok=True)
    
    print("StAMA - Statistical Analysis in Modern Academia")
    print("Script 4: Clustering Analysis")
    print("="*60)
    
    # Load and prepare data
    X, y_true, titles, publishers, feature_names = load_and_prepare_data()
    
    # Scale features
    X_standard, X_robust, scaler_standard, scaler_robust = scale_features(X)
    
    # Use robust scaling for main analysis
    X_scaled = X_robust
    
    # Determine optimal number of clusters
    optimal_k, optimization_results = determine_optimal_clusters(X_scaled)
    
    # Perform different clustering methods
    print("\n" + "="*40)
    print("CLUSTERING METHODS COMPARISON")
    print("="*40)
    
    # K-means
    kmeans_model, kmeans_labels, kmeans_silhouette = perform_kmeans_clustering(X_scaled, optimal_k, y_true)
    
    # DBSCAN
    dbscan_labels, best_eps = perform_dbscan_clustering(X_scaled, y_true)
    
    # Hierarchical
    hierarchical_model, hierarchical_labels = perform_hierarchical_clustering(X_scaled, optimal_k, y_true)
    
    # Visualizations
    print("\n" + "="*40)
    print("CREATING VISUALIZATIONS")
    print("="*40)
    
    # PCA visualizations
    pca_kmeans, X_pca_kmeans = visualize_clusters_pca(X_scaled, kmeans_labels, y_true, "K-means")
    visualize_clusters_pca(X_scaled, hierarchical_labels, y_true, "Hierarchical")
    
    # t-SNE visualizations
    visualize_clusters_tsne(X_scaled, kmeans_labels, y_true, "K-means")
    visualize_clusters_tsne(X_scaled, hierarchical_labels, y_true, "Hierarchical")
    
    # Analyze cluster characteristics (using K-means results)
    cluster_stats_df, df_clusters = analyze_cluster_characteristics(X, kmeans_labels, feature_names, titles, publishers)
    
    # Identify suspicious patterns
    suspicious_df = identify_suspicious_patterns(df_clusters, y_true)
    
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved in 'images' directory:")
    print("- optimal_clusters_analysis.png")
    print("- cluster_optimization_results.csv")
    print("- *_pca_visualization.png")
    print("- *_tsne_visualization.png")
    print("- cluster_characteristics.csv")
    print("- cluster_characteristics_heatmap.png")
    print("- suspicious_patterns_analysis.csv")

if __name__ == "__main__":
    main()
