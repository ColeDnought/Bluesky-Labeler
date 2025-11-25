from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from typing import Optional
from analysis_helpers import load_url_data, analyze_authors_comprehensive, add_domain_column, populate_follower_count


def extract_did_from_link(link: str) -> Optional[str]:
    """Extract DID from a Bluesky profile link."""
    # Match pattern like: https://bsky.app/profile/did:plc:xxxxx
    match = re.search(r'did:plc:[a-z0-9]+', link)
    if match:
        return match.group(0)
    return None


def detect_file_format(csv_file: str) -> str:
    """Detect whether the CSV is url_stream format, test_data format, or feature format."""
    df = pd.read_csv(csv_file, nrows=1)
    if 'link' in df.columns:
        return 'test_data'
    elif 'timestamp' in df.columns and 'url' in df.columns:
        return 'url_stream'
    elif 'unique_domains' in df.columns and 'unique_urls' in df.columns:
        return 'feature_data'
    else:
        raise ValueError("Unknown CSV format. Expected 'link' column (test data), 'timestamp'/'url' columns (url stream), or feature columns (feature data)")


def load_test_data(csv_file: str) -> pd.DataFrame:
    """
    Load test data CSV with profile links.
    
    Args:
        csv_file: Path to CSV with 'link' and optionally 'label' columns
        
    Returns:
        DataFrame with 'author' (DID) and optionally 'label' columns
    """
    df = pd.read_csv(csv_file)
    df['author'] = df['link'].apply(extract_did_from_link)
    
    # Drop rows where DID extraction failed
    invalid_count = df['author'].isna().sum()
    if invalid_count > 0:
        print(f"Warning: Could not extract DID from {invalid_count} links")
        df = df.dropna(subset=['author'])
    
    return df


def load_model(directory: str = 'classifier') -> tuple:
    """Load the saved classifier, scaler, and configuration."""
    directory_path = Path(directory)
    classifier = joblib.load(directory_path / 'spam_classifier.joblib')
    scaler = joblib.load(directory_path / 'feature_scaler.joblib')
    config = joblib.load(directory_path / 'classifier_config.joblib')
    return classifier, scaler, config


def prepare_features(df: pd.DataFrame, feature_columns: list, scaler) -> pd.DataFrame:
    """Prepare and normalize features for inference."""
    # Make sure all required features are present
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    
    # Extract and normalize features
    X = df[feature_columns].copy()
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)


def predict(df: pd.DataFrame, classifier, scaler, config, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Make predictions on the input dataframe.
    
    Args:
        df: DataFrame with author statistics
        classifier: Trained classifier
        scaler: Fitted scaler for feature normalization
        config: Configuration dict with feature columns and threshold
        threshold: Override threshold (uses saved threshold if None)
    
    Returns:
        DataFrame with predictions and probabilities
    """
    # Use provided threshold or fall back to config
    if threshold is None:
        threshold = config['threshold']
    
    feature_columns = config['feature_columns']
    spam_idx = config['spam_class_index']
    
    # Prepare features
    X = prepare_features(df, feature_columns, scaler)
    
    # Get probabilities
    probabilities = classifier.predict_proba(X)
    spam_proba = probabilities[:, spam_idx]
    
    # Make predictions based on threshold
    predictions = np.where(spam_proba >= threshold, 'spam', 'good')
    
    # Create results dataframe
    results = df.copy()
    results['prediction'] = predictions
    results['spam_probability'] = spam_proba
    results['threshold_used'] = threshold
    
    return results


def run_inference_on_file(input_file: str, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Run inference on a CSV file of URL data.
    
    Args:
        input_file: Path to CSV file with URL stream data
        threshold: Optional threshold override (uses config threshold if None)
    
    Returns:
        DataFrame with author statistics and predictions
    """
    # Load model
    classifier, scaler, config = load_model()
    
    # Load and process data
    print(f"Loading data from {input_file}...")
    df = load_url_data(input_file)
    df = add_domain_column(df)
    
    # Analyze authors
    author_stats = analyze_authors_comprehensive(df)
    
    # Populate follower counts if needed
    if 'followers_count' in config['feature_columns'] or 'follows_count' in config['feature_columns']:
        print("Fetching follower counts...")
        author_stats = populate_follower_count(author_stats)
    
    # Run predictions
    results = predict(author_stats, classifier, scaler, config, threshold)
    
    return results


def run_inference_on_test_data(input_file: str, url_stream_file: str = 'url_stream.csv', threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Run inference on test data CSV with profile links.
    
    This matches the preprocessing in url_analysis.ipynb:
    1. Load URL stream data to get posting history
    2. Load test data with profile links and labels
    3. Merge to get author statistics for labeled users
    4. Populate follower counts
    5. Run predictions
    
    Args:
        input_file: Path to CSV file with 'link' column (and optionally 'label')
        url_stream_file: Path to URL stream CSV with posting history
        threshold: Optional threshold override (uses config threshold if None)
    
    Returns:
        DataFrame with predictions and ground truth labels if available
    """
    # Load model
    classifier, scaler, config = load_model()
    feature_columns = config['feature_columns']
    
    # Step 1: Load URL stream data (same as notebook)
    print(f"Loading URL stream data from {url_stream_file}...")
    df = load_url_data(url_stream_file)
    df = add_domain_column(df)
    
    # Step 2: Load test data with labels
    print(f"Loading test data from {input_file}...")
    labeled = load_test_data(input_file)
    print(f"Found {len(labeled)} labeled authors")
    
    # Step 3: Analyze authors and merge with labels (same as notebook)
    print("Analyzing author statistics...")
    author_stats = analyze_authors_comprehensive(df, labels_df=labeled)
    
    # Filter to only labeled authors (those in test_data.csv)
    test_data = author_stats[author_stats['label'].notnull()].copy()
    print(f"Found {len(test_data)} authors with both posting history and labels")
    
    if len(test_data) == 0:
        print("Warning: No authors from test data found in URL stream. Falling back to profile-only inference.")
        return run_inference_profile_only(input_file, threshold)
    
    # Step 4: Populate follower counts if needed
    if 'followers_count' in feature_columns or 'follows_count' in feature_columns:
        print("Fetching follower counts...")
        test_data = populate_follower_count(test_data)
    
    # Step 5: Run predictions
    results = predict(test_data, classifier, scaler, config, threshold)
    
    return results


def run_inference_profile_only(input_file: str, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Fallback inference using only profile data (no URL posting history).
    
    This is used when test data authors are not found in url_stream.csv.
    Warning: Results may be less accurate without posting history features.
    
    Args:
        input_file: Path to CSV file with 'link' column
        threshold: Optional threshold override (uses config threshold if None)
    """
    import asyncio
    from fetch_users import get_follower_ratios
    
    classifier, scaler, config = load_model()
    feature_columns = config['feature_columns']
    
    print(f"Loading test data from {input_file}...")
    df = load_test_data(input_file)
    print(f"Found {len(df)} unique authors")
    
    # Set URL-based features to 0 (not available without posting history)
    for col in feature_columns:
        if col in ['unique_domains', 'unique_urls', 'total_posts']:
            df[col] = 0
            print(f"Warning: '{col}' requires URL stream data. Setting to 0.")
    
    # Fetch follower data
    print("Fetching follower counts from Bluesky API...")
    authors = df['author'].tolist()
    follower_data = asyncio.run(get_follower_ratios(authors))
    
    df = df.merge(
        follower_data[['did', 'followers_count', 'follows_count', 'follower_following_ratio']],
        left_on='author',
        right_on='did',
        how='left'
    )
    if 'did' in df.columns:
        df = df.drop(columns=['did'])
    
    for col in ['followers_count', 'follows_count', 'follower_following_ratio']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return predict(df, classifier, scaler, config, threshold)


def classify_authors(author_stats: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Classify authors from pre-computed statistics.
    
    Args:
        author_stats: DataFrame with pre-computed author statistics
        threshold: Optional threshold override (uses config threshold if None)
    
    Returns:
        DataFrame with predictions
    """
    classifier, scaler, config = load_model()
    return predict(author_stats, classifier, scaler, config, threshold)


def run_inference_on_feature_data(input_file: str, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Run inference on a CSV file that already has feature columns.
    
    This handles files like test_split.csv that have pre-computed features:
    unique_domains, unique_urls, followers_count, follows_count, and optionally label.
    
    Args:
        input_file: Path to CSV file with feature columns
        threshold: Optional threshold override (uses config threshold if None)
    
    Returns:
        DataFrame with predictions
    """
    # Load model
    classifier, scaler, config = load_model()
    
    # Load data
    print(f"Loading feature data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Found {len(df)} samples")
    
    # Run predictions
    results = predict(df, classifier, scaler, config, threshold)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spam classification on URL data or test data')
    parser.add_argument('input_file', nargs='?', default='url_stream.csv',
                        help='Input CSV file (url_stream format, test_data format with profile links, or feature data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override the classification threshold (0.0-1.0). If not provided, uses the threshold from the saved config.')
    
    args = parser.parse_args()
    
    if args.threshold is not None:
        print(f"Using custom threshold: {args.threshold}")
    
    # Detect file format and run appropriate inference
    file_format = detect_file_format(args.input_file)
    print(f"Detected file format: {file_format}")
    
    if file_format == 'test_data':
        results = run_inference_on_test_data(args.input_file, threshold=args.threshold)
    elif file_format == 'feature_data':
        results = run_inference_on_feature_data(args.input_file, threshold=args.threshold)
    else:
        results = run_inference_on_file(args.input_file, threshold=args.threshold)
    
    # Display summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Total authors: {len(results)}")
    print(f"Predicted spam: {(results['prediction'] == 'spam').sum()}")
    print(f"Predicted good: {(results['prediction'] == 'good').sum()}")
    print(f"Threshold used: {results['threshold_used'].iloc[0]}")
    
    # Show classification report if ground truth labels are available
    if 'label' in results.columns:       
        print(f"\nClassification Report (threshold = {results['threshold_used'].iloc[0]})")
        print("=" * 50)
        print(classification_report(results['label'], results['prediction'], target_names=['good', 'spam'], zero_division=1))
    
    # Save results if output specified
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
