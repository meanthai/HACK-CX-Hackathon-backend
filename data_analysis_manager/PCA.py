import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_feature_importance_and_pca(data_path, target_column=None, analysis_type='classification'):
    """
    Analyze feature importance and apply PCA to the dataset
    
    Parameters:
    data_path: path to the CSV file
    target_column: name of the target column (if None, will analyze all features)
    analysis_type: 'classification' or 'regression'
    """
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return
    
    # Prepare the data
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_categorical = df.select_dtypes(exclude=[np.number]).copy()
    
    # Handle categorical variables if any
    if not df_categorical.empty:
        le = LabelEncoder()
        for col in df_categorical.columns:
            if col != target_column:  # Don't encode the target if it's categorical
                df_categorical[col] = le.fit_transform(df_categorical[col].astype(str))
        
        # Combine numeric and encoded categorical data
        df_processed = pd.concat([df_numeric, df_categorical], axis=1)
    else:
        df_processed = df_numeric.copy()
    
    # Remove target column from features if specified
    if target_column and target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        features_for_analysis = X
    else:
        # If no target specified, analyze all features
        X = df_processed
        y = None
        features_for_analysis = df_processed
    
    # Standardize the features for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Importance Analysis (if target is available)
    if y is not None:
        if analysis_type == 'classification':
            # Random Forest Feature Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # Mutual Information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
        else:  # regression
            # Random Forest Feature Importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # Mutual Information
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Plot Random Forest Feature Importance
        plt.subplot(2, 3, 1)
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importance_df.head(10), y='feature', x='importance')
        plt.title('Random Forest Feature Importance (Top 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        
        # Plot Mutual Information
        plt.subplot(2, 3, 2)
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        sns.barplot(data=mi_df.head(10), y='feature', x='mi_score')
        plt.title('Mutual Information Scores (Top 10)', fontsize=14, fontweight='bold')
        plt.xlabel('MI Score')
    
    # 2. Correlation Heatmap
    plt.subplot(2, 3, 3)
    correlation_matrix = X.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 3. PCA Analysis
    # Determine optimal number of components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Plot explained variance ratio
    plt.subplot(2, 3, 4)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Explained Variance Ratio', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find number of components for 90% and 95% variance
    n_components_90 = np.argmax(cumsum_var >= 0.90) + 1
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    print(f"\nPCA Analysis:")
    print(f"Components needed for 90% variance: {n_components_90}")
    print(f"Components needed for 95% variance: {n_components_95}")
    
    # 4. PCA with optimal components (90% variance)
    pca_optimal = PCA(n_components=min(n_components_90, len(X.columns)))
    X_pca = pca_optimal.fit_transform(X_scaled)
    
    # Plot first two principal components
    plt.subplot(2, 3, 5)
    if y is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'First Two Principal Components\n(Colored by {target_column})', 
                 fontsize=14, fontweight='bold')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        plt.title('First Two Principal Components', fontsize=14, fontweight='bold')
    
    plt.xlabel(f'PC1 ({pca_optimal.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_optimal.explained_variance_ratio_[1]:.2%} variance)')
    
    # 5. PCA Component Composition
    plt.subplot(2, 3, 6)
    # Show which original features contribute most to first few components
    components_df = pd.DataFrame(
        pca_optimal.components_[:min(3, pca_optimal.n_components_)].T,
        columns=[f'PC{i+1}' for i in range(min(3, pca_optimal.n_components_))],
        index=X.columns
    )
    
    sns.heatmap(components_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('PCA Component Composition\n(Feature Loadings)', fontsize=14, fontweight='bold')
    plt.ylabel('Original Features')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n{'='*50}")
    print("FEATURE ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    if y is not None:
        print(f"\nTop 5 Most Important Features (Random Forest):")
        for i, (feature, importance) in enumerate(importance_df.head().values):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        print(f"\nTop 5 Most Important Features (Mutual Information):")
        for i, (feature, mi_score) in enumerate(mi_df.head().values):
            print(f"{i+1}. {feature}: {mi_score:.4f}")
    
    print(f"\nPCA Results:")
    print(f"Original number of features: {X.shape[1]}")
    print(f"Reduced to {pca_optimal.n_components_} components")
    print(f"Explained variance: {sum(pca_optimal.explained_variance_ratio_):.2%}")
    
    print(f"\nExplained variance by component:")
    for i, var_ratio in enumerate(pca_optimal.explained_variance_ratio_):
        print(f"PC{i+1}: {var_ratio:.2%}")
    
    # Return processed data and models for further use
    return {
        'original_data': df,
        'processed_features': X,
        'scaled_features': X_scaled,
        'pca_features': X_pca,
        'pca_model': pca_optimal,
        'scaler': scaler,
        'feature_importance': importance_df if y is not None else None,
        'mutual_info': mi_df if y is not None else None
    }

# Example usage for your data:

# For Credit Score Analysis (assuming you have a target column)
print("Analyzing Credit Score Data...")
try:
    credit_results = analyze_feature_importance_and_pca(
        "data_analysis_manager/processed_data/user_data_credit_score.csv",
        target_column=None,  # Set to your target column name if available
        analysis_type='regression'
    )
except:
    print("Credit score data file not found or error in processing")

print("\n" + "="*60 + "\n")

# For NBO Analysis (classification)
print("Analyzing NBO Data...")
try:
    nbo_results = analyze_feature_importance_and_pca(
        "data_analysis_manager/processed_data/user_data_credit_score.csv",
        target_column='category',  # Product category as target
        analysis_type='classification'
    )
except:
    print("NBO data file not found or error in processing")

# Additional function for interactive analysis
def quick_pca_analysis(data, n_components=2):
    """
    Quick PCA analysis for any dataset
    """
    # Prepare data
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data_pca,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    print(f"PCA completed. Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
    
    return pca_df, pca, scaler
