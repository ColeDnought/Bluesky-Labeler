"""Script for testing the automated labeler"""

import argparse
import json
import os

import pandas as pd
from atproto import Client
from dotenv import load_dotenv

from pylabel import AutomatedLabeler, label_post, did_from_handle

load_dotenv(override=True)
USERNAME = os.getenv("BSKY_USR")
PW = os.getenv("BSKY_PWD")

def main():
    """
    Main function for the test script
    """
    client = Client()
    labeler_client = None
    client.login(USERNAME, PW)
    did = did_from_handle(USERNAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("labeler_inputs_dir", type=str)
    parser.add_argument("input_urls", type=str)
    parser.add_argument("--emit_labels", action="store_true")
    args = parser.parse_args()

    if args.emit_labels:
        labeler_client = client.with_proxy("atproto_labeler", did)

    labeler = AutomatedLabeler(client, args.labeler_inputs_dir)

    urls = pd.read_csv(args.input_urls)
    num_correct, total = 0, urls.shape[0]
    
    # Track metrics per label: {label: {'tp': count, 'fp': count, 'fn': count}}
    label_metrics = {}
    
    for _index, row in urls.iterrows():
        url, expected_labels = row["URL"], json.loads(row["Labels"])
        labels = labeler.moderate_post(url)
        print(f"For {url}, labeler produced {labels}")
        
        # Convert to sets for easier comparison
        expected_set = set(expected_labels)
        predicted_set = set(labels)
        
        # Update per-label metrics
        # True positives: labels that are in both expected and predicted
        for label in predicted_set & expected_set:
            if label not in label_metrics:
                label_metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
            label_metrics[label]['tp'] += 1
        
        # False positives: labels predicted but not expected
        for label in predicted_set - expected_set:
            if label not in label_metrics:
                label_metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
            label_metrics[label]['fp'] += 1
        
        # False negatives: labels expected but not predicted
        for label in expected_set - predicted_set:
            if label not in label_metrics:
                label_metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
            label_metrics[label]['fn'] += 1
        
        if sorted(labels) == sorted(expected_labels):
            num_correct += 1
        else:
            print(f"MISMATCH: For {url}, labeler produced {labels}, expected {expected_labels}")
        if args.emit_labels and (len(labels) > 0):
            label_post(client, labeler_client, url, labels)
    
    # Calculate and display precision/recall per label
    print("\n" + "="*60)
    print("METRICS PER LABEL")
    print("="*60)
    
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    
    for label in sorted(label_metrics.keys()):
        tp = label_metrics[label]['tp']
        fp = label_metrics[label]['fp']
        fn = label_metrics[label]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        print(f"\nLabel: {label}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
    
    # Overall metrics (micro-averaged)
    print("\n" + "="*60)
    print("OVERALL METRICS (Micro-averaged)")
    print("="*60)
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    print(f"Total TP: {overall_tp}, FP: {overall_fp}, FN: {overall_fn}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1 Score: {overall_f1:.3f}")
    
    print("\n" + "="*60)
    print(f"The labeler produced {num_correct} correct labels assignments out of {total}")
    print(f"Overall ratio of correct label assignments: {num_correct/total:.3f}")


if __name__ == "__main__":
    main()
