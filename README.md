# Bluesky Spam Account Classifier
### Team 8 | Sara Frazer, Om Kamath, Cole Donat


## Testing

_Don't forget to install the requirements from_ `requirements.txt` _before running tests._

To view test outputs in terminal: `python policy_proposal_labeler.py`
- This will load the test data from `data/test_data.csv`, run the classifier, and print performance statistics.

Tests can also be run on the **Run Classifier** page of the Streamlit app: `streamlit run app.py`

## Running beyond testing
The classifier itself is implemented in a scikit-learn decision tree model, with a normalizing scaler. You can run it by loading the model and scaler from the `classifier/` directory using joblib.

```python
from analysis_helpers import load_model

# Load model and scaler
classifier, scaler, config = load_model()

# Scale features
normalized_data = scaler.transform(raw_data)

# Run predictions
probs = classifier.predict_proba(normalized_data)
```

You can also find an example in `policy_proposal_labeler.py`.


## Data
The data in `data/test_data.csv` is preprocessed from the Firehose data in `data/url_stream.csv` using the same feature extraction as in the main app. You may process the data from user links by using the `analysis_helpers` functions `load_url_data`, `analyze_authors_comprehensive`, and `add_domain_column`.

## Repository Structure
- `policy_proposal_labeler.py`: Script for running tests on the spam classifier
- `app.py`: Main Streamlit app for viewing and classifying Bluesky accounts
- `fetch_users.py`: Used to cross-reference Firehose data with user profile data from Bluesky API
- `analysis_helpers.py`: Helper functions for data analysis and visualization
- `classifier/`: Directory containing the trained spam classifier model and scaler
- `data/`: Directory containing datasets used for training and testing
    - `data/url_stream.csv` is the raw data stream from the Firehose API
- `old_files/`: Contains scripts used for data aggregation/analysis, not used for inference time
- `.env`: Environment variables for Bluesky API access (a burner account)