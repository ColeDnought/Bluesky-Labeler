import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import asyncio
from fetch_users import get_follower_ratios

st.set_page_config(page_title="Firehose URL Dashboard", layout="wide")

st.title("Firehose URL Stream Dashboard")

@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], low_memory=False)
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamp'])
        return df
    except FileNotFoundError:
        return None

def get_domain(url):
    try:
        if not isinstance(url, str):
            return "Invalid"
        # Handle cases where url might not have scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return "Error"

def ensure_scheme(u):
    try:
        if not isinstance(u, str):
            return u
        if u.startswith(('http://', 'https://')):
            return u
        return 'http://' + u
    except Exception:
        return u

@st.cache_data(ttl=300)
def enrich_with_user_data(df):
    """Enrich dataframe with user handles, profile URLs, and follower ratios for all unique authors"""
    unique_authors = df['author'].unique()
    
    try:
        # Fetch all user data including follower ratios in one call
        user_data_df = asyncio.run(get_follower_ratios(list(unique_authors)))
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
        # Return a fallback dataframe
        user_data_df = pd.DataFrame({
            'did': list(unique_authors),
            'handle': list(unique_authors),
            'profile_url': [f"https://bsky.app/profile/{did}" for did in unique_authors],
            'followers_count': [None] * len(unique_authors),
            'follows_count': [None] * len(unique_authors),
            'follower_following_ratio': [None] * len(unique_authors)
        })
    
    # Merge user info into the main dataframe
    df_enriched = df.merge(
        user_data_df[['did', 'handle', 'profile_url', 'follower_following_ratio']],
        left_on='author',
        right_on='did',
        how='left'
    )
    return df_enriched

def show_general_stats(df):
    st.header("General Statistics")
    st.write(f"Total Records: {len(df)}")

    # Top Domains
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Most Common Domains")
        top_domains_series = df['domain'].value_counts().head(20)
        
        # Calculate unique authors per domain
        domain_authors = df.groupby('domain')['author'].nunique().to_dict()
        
        top_domains_df = pd.DataFrame({
            'domain': top_domains_series.index.astype(str),
            'count': top_domains_series.values
        })
        top_domains_df['unique_authors'] = top_domains_df['domain'].map(domain_authors)
        
        st.bar_chart(top_domains_df.set_index('domain')['count'])

    with col2:
        st.write("Top 20 Domains Count")
        st.dataframe(top_domains_df[['domain', 'count', 'unique_authors']])

    # Top Pages
    st.divider()
    st.subheader("Most Common Pages")
    
    # Filter by domain
    domain_options = ['All'] + list(top_domains_df['domain'])
    selected_domain = st.selectbox("Filter by domain:", domain_options)

    if selected_domain != 'All':
        filtered_df = df[df['domain'] == selected_domain]
        st.write(f"Top Pages in {selected_domain}")
    else:
        filtered_df = df
        st.write("Top Pages (All Domains)")

    top_pages_series = filtered_df['url'].value_counts().head(20)
    top_pages_df = pd.DataFrame({
        'url': top_pages_series.index.astype(str),
        'count': top_pages_series.values
    })

    top_pages_df['url'] = top_pages_df['url'].apply(ensure_scheme)

    # Display pages with clickable links and counts using LinkColumn
    st.dataframe(
        top_pages_df,
        column_config={
            'url': st.column_config.LinkColumn('URL'),
            'count': st.column_config.NumberColumn('Count')
        }
    )
    
    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df)

def show_suspicious_authors(df):
    st.header("Suspicious Authors Analysis")
    st.markdown("""
    This page identifies authors who post frequently, often linking to the same domain.
    **Criteria:**
    - At least 5 posts
    - At least 60% of posts are to the same domain
    """)

    # 1. Calculate stats per author
    author_stats = df.groupby('author').agg(
        total_posts=('url', 'count'),
        unique_domains=('domain', 'nunique')
    ).reset_index()

    # 2. Find top domain per author
    top_domains = df.groupby(['author', 'domain']).size().reset_index(name='domain_count')
    top_domains = top_domains.sort_values(['author', 'domain_count'], ascending=[True, False])
    top_domains = top_domains.groupby('author').first().reset_index()

    # 3. Merge stats
    author_analysis = pd.merge(author_stats, top_domains, on='author')
    author_analysis['domain_share'] = author_analysis['domain_count'] / author_analysis['total_posts']

    # 4. Filter for suspicious authors
    suspicious_authors = author_analysis[
        (author_analysis['total_posts'] >= 1) & 
        (author_analysis['domain_share'] >= 0.0)
    ].copy()

    if suspicious_authors.empty:
        st.warning("No suspicious authors found with current criteria.")
        return

    # 5. Calculate time period and frequency
    suspicious_posts = df[df['author'].isin(suspicious_authors['author'])]
    
    time_stats = suspicious_posts.groupby('author')['timestamp'].agg(['min', 'max']).reset_index()
    time_stats['duration'] = time_stats['max'] - time_stats['min']
    time_stats['duration_seconds'] = time_stats['duration'].dt.total_seconds()

    suspicious_authors = pd.merge(suspicious_authors, time_stats, on='author')
    
    # Calculate frequency (posts per minute)
    suspicious_authors['posts_per_minute'] = suspicious_authors['total_posts'] / ((suspicious_authors['duration_seconds'] / 60) + 0.001)
    
    suspicious_authors = suspicious_authors.sort_values('posts_per_minute', ascending=False)

    # Merge in the handle, profile_url, and follower_following_ratio from the enriched df
    suspicious_authors = suspicious_authors.merge(
        df[['author', 'handle', 'profile_url', 'follower_following_ratio']].drop_duplicates('author'),
        on='author',
        how='left'
    )

    st.subheader(f"Found {len(suspicious_authors)} Suspicious Authors")
    st.write("Click on a row to view details below.")
    
    # Display table with selection - create a display dataframe with handle shown
    display_df = suspicious_authors[['handle', 'domain', 'total_posts', 'duration', 'posts_per_minute', 'follower_following_ratio']].copy()
    display_df['profile_link'] = suspicious_authors['profile_url']
    
    event = st.dataframe(
        display_df[['profile_link', 'domain', 'total_posts', 'duration', 'posts_per_minute', 'follower_following_ratio']],
        column_config={
            'profile_link': st.column_config.LinkColumn('Profile'),
            'domain': 'Top Domain',
            'total_posts': 'Posts',
            'duration': 'Duration',
            'posts_per_minute': st.column_config.NumberColumn('Posts/Min', format="%.2f"),
            'follower_following_ratio': st.column_config.NumberColumn('Follower:Following', format="%.2f")
        },
        on_select="rerun",
        selection_mode="single-row"
    )

    # Show details when a row is selected
    if event.get("selection") and event["selection"].get("rows"):
        selected_idx = event["selection"]["rows"][0]
        selected_author = suspicious_authors.iloc[selected_idx]['author']
        author_domain = suspicious_authors.iloc[selected_idx]['domain']
        
        st.divider()
        st.subheader(f"Details for {selected_author}")
        
        author_posts = df[df['author'] == selected_author].sort_values('timestamp', ascending=False)
        campaign_posts = author_posts[author_posts['domain'] == author_domain]
        
        # Get unique URLs with their counts and sample text
        campaign_urls = campaign_posts.groupby('url').agg(
            count=('url', 'size'),
            text=('text', 'first')
        ).reset_index()
        
        campaign_urls = campaign_urls.sort_values('count', ascending=False)
        campaign_urls['url_link'] = campaign_urls['url'].apply(ensure_scheme)
        
        st.write(f"**{len(campaign_urls)} unique URLs** from **{author_domain}** ({campaign_urls['count'].sum()} total posts)")
        
        st.dataframe(
            campaign_urls[['url_link', 'count', 'text']],
            column_config={
                'url_link': st.column_config.LinkColumn('URL'),
                'count': st.column_config.NumberColumn('Times Posted'),
                'text': st.column_config.TextColumn('Post Text')
            }
        )

def show_threshold_tuning():
    """Interactive threshold tuning page with TPR/FPR metrics"""
    st.header("Threshold Tuning")
    st.markdown("""
    Adjust feature thresholds to tune spam detection. Authors exceeding thresholds on normalized features
    will be flagged as suspicious. The metrics show performance on a labeled test set.
    """)
    
    # Load labeled test data
    try:
        labeled_df = pd.read_csv('test_data.csv')
        labeled_df['author'] = labeled_df['link'].apply(lambda x: x.split('/')[-1])
    except FileNotFoundError:
        st.error("test_data.csv not found. Please create a labeled test set first.")
        return
    
    # Load and prepare author stats
    from analysis_helpers import load_url_data, add_domain_column, analyze_authors_comprehensive
    
    @st.cache_data
    def get_author_stats():
        df = load_url_data('url_stream.csv')
        df = add_domain_column(df)
        stats = analyze_authors_comprehensive(df, labels_df=labeled_df)
        return stats
    
    with st.spinner('Loading author statistics...'):
        author_stats = get_author_stats()
    
    # Fetch follower data separately using the same async pattern as enrich_with_user_data
    @st.cache_data(ttl=300)
    def get_follower_data_for_authors(authors_list):
        try:
            return asyncio.run(get_follower_ratios(authors_list))
        except RuntimeError:
            # Event loop already running - use nest_asyncio workaround
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(get_follower_ratios(authors_list))
    
    # Only fetch follower data for labeled authors (smaller set)
    labeled_authors = author_stats[author_stats['label'].notnull()]['author'].unique().tolist()
    
    if labeled_authors:
        with st.spinner('Fetching follower data...'):
            try:
                follower_df = get_follower_data_for_authors(tuple(labeled_authors))
                author_stats = author_stats.merge(
                    follower_df[['did', 'followers_count', 'follows_count', 'follower_following_ratio']],
                    left_on='author',
                    right_on='did',
                    how='left'
                )
                if 'did' in author_stats.columns:
                    author_stats = author_stats.drop(columns=['did'])
            except Exception as e:
                st.warning(f"Could not fetch follower data: {e}")
    
    # Filter to only labeled data for evaluation
    test_data = author_stats[author_stats['label'].notnull()].copy()
    
    if test_data.empty:
        st.warning("No labeled authors found in the dataset.")
        return
    
    # Define feature columns
    feature_columns = ['unique_domains', 'unique_urls', 'avg_time_between_posts', 
                       'followers_count', 'follows_count', 'follower_following_ratio']
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in test_data.columns]
    
    if not available_features:
        st.error("No feature columns available in the data.")
        return
    
    # Normalize features for the test data
    from sklearn.preprocessing import StandardScaler
    
    # Handle missing values
    test_data_clean = test_data.dropna(subset=available_features)
    
    if len(test_data_clean) < len(test_data):
        st.info(f"Dropped {len(test_data) - len(test_data_clean)} rows with missing feature values.")
    
    scaler = StandardScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(test_data_clean[available_features]),
        columns=available_features,
        index=test_data_clean.index
    )
    
    st.subheader("Feature Thresholds")
    st.markdown("Set thresholds for normalized features (z-scores). Positive = above mean, Negative = below mean.")
    
    # Create sliders for each feature
    thresholds = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(available_features):
        with cols[i % 2]:
            # Different default directions based on feature semantics
            if feature in ['followers_count', 'follower_following_ratio']:
                # Lower values are suspicious
                default_val = -1.0
                help_text = f"Flag if below threshold (lower {feature} = more suspicious)"
            else:
                # Higher values are suspicious  
                default_val = 1.0
                help_text = f"Flag if above threshold (higher {feature} = more suspicious)"
            
            thresholds[feature] = st.slider(
                f"{feature}",
                min_value=-3.0,
                max_value=3.0,
                value=default_val,
                step=0.1,
                help=help_text
            )
    
    # Checkbox for threshold direction
    st.subheader("Threshold Direction")
    threshold_directions = {}
    cols2 = st.columns(2)
    
    for i, feature in enumerate(available_features):
        with cols2[i % 2]:
            default_above = feature not in ['followers_count', 'follower_following_ratio']
            threshold_directions[feature] = st.checkbox(
                f"{feature}: Flag if ABOVE threshold",
                value=default_above,
                key=f"dir_{feature}"
            )
    
    # Calculate predictions based on thresholds
    predictions = pd.Series([False] * len(normalized_features), index=normalized_features.index)
    
    for feature in available_features:
        if threshold_directions[feature]:
            # Flag if above threshold
            predictions = predictions | (normalized_features[feature] > thresholds[feature])
        else:
            # Flag if below threshold
            predictions = predictions | (normalized_features[feature] < thresholds[feature])
    
    # Get ground truth (label 'bad' or 'spam' is the positive class)
    ground_truth = test_data_clean['label'].apply(
        lambda x: x.lower() in ['spam', 'bad'] if isinstance(x, str) else False
    )
    
    # Calculate metrics
    true_positives = (predictions & ground_truth).sum()
    false_positives = (predictions & ~ground_truth).sum()
    true_negatives = (~predictions & ~ground_truth).sum()
    false_negatives = (~predictions & ground_truth).sum()
    
    total_positives = ground_truth.sum()
    total_negatives = (~ground_truth).sum()
    
    tpr = true_positives / total_positives if total_positives > 0 else 0
    fpr = false_positives / total_negatives if total_negatives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(predictions) if len(predictions) > 0 else 0
    
    # Display metrics
    st.divider()
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("True Positive Rate (Recall)", f"{tpr:.2%}")
        st.caption(f"{true_positives}/{total_positives} spam detected")
    
    with col2:
        st.metric("False Positive Rate", f"{fpr:.2%}")
        st.caption(f"{false_positives}/{total_negatives} legitimate flagged")
    
    with col3:
        st.metric("Precision", f"{precision:.2%}")
        st.caption(f"{true_positives}/{true_positives + false_positives} flagged are spam")
    
    with col4:
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.caption(f"{true_positives + true_negatives}/{len(predictions)} correct")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    confusion_df = pd.DataFrame({
        'Predicted Spam': [true_positives, false_positives],
        'Predicted Legitimate': [false_negatives, true_negatives]
    }, index=['Actual Spam', 'Actual Legitimate'])
    
    st.dataframe(confusion_df, use_container_width=True)
    
    # Show flagged authors
    st.divider()
    st.subheader("Flagged Authors")
    
    flagged_authors = test_data_clean[predictions].copy()
    flagged_authors['actual_label'] = ground_truth[predictions].map({True: 'Spam', False: 'Legitimate'})
    
    if not flagged_authors.empty:
        display_cols = ['author', 'actual_label'] + available_features[:4]
        display_cols = [c for c in display_cols if c in flagged_authors.columns]
        st.dataframe(flagged_authors[display_cols].head(20))
    else:
        st.info("No authors flagged with current thresholds.")

# Main App Logic
CSV_FILE = 'url_stream.csv'
df = load_data(CSV_FILE)

if df is None:
    st.error(f"File {CSV_FILE} not found. Please run the firehose script first.")
else:
    # Apply domain extraction globally
    if 'domain' not in df.columns:
        df['domain'] = df['url'].apply(get_domain)
    
    # Enrich with user data (handles, profile URLs, follower ratios) once at the start
    if 'handle' not in df.columns:
        with st.spinner('Fetching user data...'):
            df = enrich_with_user_data(df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["General Stats", "Suspicious Authors", "Threshold Tuning"])

    if page == "General Stats":
        show_general_stats(df)
    elif page == "Suspicious Authors":
        show_suspicious_authors(df)
    elif page == "Threshold Tuning":
        show_threshold_tuning()
