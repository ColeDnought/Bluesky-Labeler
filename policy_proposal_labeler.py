from warnings import filterwarnings
filterwarnings("ignore", module="pydantic")
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.metrics import precision_score, recall_score, classification_report

def load_model(directory: str = 'classifier') -> tuple:
    """Load the saved classifier, scaler, and configuration."""
    directory_path = Path(directory)
    classifier = joblib.load(directory_path / 'spam_classifier.joblib')
    scaler = joblib.load(directory_path / 'feature_scaler.joblib')
    config = joblib.load(directory_path / 'classifier_config.joblib')
    return classifier, scaler, config


def print_spam_metrics(results: pd.DataFrame, labels: pd.Series) -> None:
    """
    Print precision and recall for spam label only.
    
    Args:
        results: DataFrame with 'prediction' and 'label' columns
        labels: Series with true labels
    """
    
    print(classification_report(labels, results['prediction'], zero_division=1, output_dict=True))


def run_inference(df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
    """
    Gets predictions for spam classification. Does not apply thresholding.

    Args:
        input_file: Path to CSV file with feature columns
        threshold: Optional threshold override (uses config threshold if None)
    
    Returns:
        Tuple containing precision and recall for spam label
    """
    # Load model
    classifier, scaler, config = load_model()

    # Scale features
    X_test = scaler.transform(df[config['feature_columns']])
    
    # Run predictions
    probs = classifier.predict_proba(X_test)

    # Apply threshold
    threshold = threshold if threshold is not None else config['threshold']
    classes = list(classifier.classes_)
    spam_idx = classes.index('spam')

    return np.where(probs[:, spam_idx] >= threshold, 'spam', 'good')

def compute_stats(results: np.ndarray, labels: pd.Series) -> tuple[float, float]:
    """
    Compute precision and recall for spam label only.
    
    Args:
        results: DataFrame with 'prediction' and 'label' columns

    Returns:
        Tuple containing precision and recall for spam label
    """
    precision = precision_score(labels, results, pos_label='spam', zero_division=1)
    recall = recall_score(labels, results, pos_label='spam', zero_division=1)
    
    return precision, recall


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run spam classification on URL data or test data')
    parser.add_argument('input_file', nargs='?', default='data/test_data.csv',
                        help='Input CSV file with feature columns (unique_domains, unique_urls, etc.)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override the classification threshold (0.0-1.0). If not provided, uses the threshold from the saved config.')
    
    args = parser.parse_args()
    
    if args.threshold is not None:
        print(f"Using custom threshold: {args.threshold}")
    
    df = pd.read_csv(args.input_file)
    results = run_inference(df)

    # Show
    print(classification_report(df['label'], results, zero_division=1))
