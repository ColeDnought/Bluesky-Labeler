from warnings import filterwarnings
filterwarnings("ignore", module="pydantic")
import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import asyncio
from fetch_users import get_follower_ratios

TITLE = "Bluesky Spam Detection Dashboard"

st.set_page_config(page_title=TITLE, layout="wide")

st.title(TITLE)
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
    Adjust the criteria below to tune the detection sensitivity.
    """)
    
    # User-adjustable criteria
    col1, col2 = st.columns(2)
    with col1:
        min_posts = st.slider(
            "Minimum number of posts",
            min_value=1,
            max_value=50,
            value=5,
            help="Authors must have at least this many posts to be considered suspicious"
        )
    with col2:
        min_domain_share = st.slider(
            "Minimum % posts to same domain",
            min_value=0,
            max_value=100,
            value=60,
            help="Authors must have at least this percentage of posts to the same domain"
        )
    
    st.divider()

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

    # 4. Filter for suspicious authors using dynamic criteria
    suspicious_authors = author_analysis[
        (author_analysis['total_posts'] >= min_posts) & 
        (author_analysis['domain_share'] >= min_domain_share / 100.0)
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

def show_classifier_page():
    """Page for running the trained classifier on uploaded CSV files"""
    st.header("Spam Classifier")
    st.markdown("""
    Run spam classification using the trained decision tree model.
    
    **Supported file formats:**
    - **Feature data**: CSV with columns `unique_domains`, `unique_urls`, `followers_count`, `follows_count` (and optionally `label`)
    - **Test data**: CSV with `link` column containing Bluesky profile URLs (e.g., `https://bsky.app/profile/did:plc:xxx`)
    """)
    
    # Import classifier functions
    from run_tests import (
        load_model, 
        run_inference_on_feature_data,
    )
    
    # Threshold slider
    st.subheader("Classification Settings")
    
    try:
        _, _, config = load_model()
        default_threshold = config.get('threshold', 0.5)
    except Exception:
        default_threshold = 0.5
    
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.05,
        help="Higher threshold = more conservative (fewer false positives, but may miss some spam)"
    )
    
    # Data source selection
    st.subheader("Data Source")
    
    default_test_file = 'data/test_data.csv'
    use_default = st.checkbox(
        f"Use default test file (`{default_test_file}`)", 
        value=True,
        help="Uncheck to upload a custom CSV file"
    )
    
    # File uploader (only shown if not using default)
    uploaded_file = None
    if not use_default:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with feature data or profile links"
        )
    
    # Determine which file to use
    if use_default:
        # Use default test file
        import os
        if not os.path.exists(default_test_file):
            st.error(f"Default test file `{default_test_file}` not found.")
            return
        
        df = pd.read_csv(default_test_file)
        
        st.write(f"**Using:** `{default_test_file}`")
        st.write(f"**Rows:** {len(df)}, **Columns:** {list(df.columns)}")
        
        # Show preview
        with st.expander("Preview data"):
            st.dataframe(df.head(10))
        
        # Detect file format
        if 'unique_domains' in df.columns and 'unique_urls' in df.columns:
            file_format = 'feature_data'
        elif 'link' in df.columns:
            file_format = 'test_data'
        else:
            st.error("Unknown file format in default test file.")
            return
        
        st.info(f"Detected format: **{file_format}**")
        
        # Run classification button
        if st.button("🚀 Run Classification", type="primary"):
            with st.spinner("Running classification..."):
                try:
                    precision, recall = run_inference_on_feature_data(default_test_file, threshold=threshold)
                    
                    # Display results
                    st.divider()
                    st.subheader("Classification Results")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Threshold Used", f"{threshold:.2f}")
                    with col2:
                        st.metric("Spam Precision", f"{precision:.1%}")
                    with col3:
                        st.metric("Spam Recall", f"{recall:.1%}")
                    
                except Exception as e:
                    st.error(f"Error running classification: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.write(f"**Uploaded file:** {uploaded_file.name}")
            st.write(f"**Rows:** {len(df)}, **Columns:** {list(df.columns)}")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                st.dataframe(df.head(10))
            
            # Detect file format
            if 'unique_domains' in df.columns and 'unique_urls' in df.columns:
                file_format = 'feature_data'
            else:
                st.error("Unknown file format. Please upload a CSV with feature columns (unique_domains, unique_urls, followers_count, follows_count, label)")
                return
            
            st.info(f"Detected format: **{file_format}**")
            
            # Run classification button
            if st.button("🚀 Run Classification", type="primary"):
                with st.spinner("Running classification..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                            df.to_csv(tmp.name, index=False)
                            tmp_path = tmp.name
                        
                        try:
                            precision, recall = run_inference_on_feature_data(tmp_path, threshold=threshold)
                        finally:
                            os.unlink(tmp_path)
                        
                        # Display results
                        st.divider()
                        st.subheader("Classification Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Threshold Used", f"{threshold:.2f}")
                        with col2:
                            st.metric("Spam Precision", f"{precision:.1%}")
                        with col3:
                            st.metric("Spam Recall", f"{recall:.1%}")
                        
                    except Exception as e:
                        st.error(f"Error running classification: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Example section
    st.divider()
    st.subheader("Example Files")
    st.markdown("""
    **Feature data format** (`data/test_data.csv`):
    ```csv
    unique_domains,unique_urls,avg_time_between_posts,followers_count,follows_count,follower_following_ratio,label
    1,5,8935.7,0.1,18437.6,0.05,good
    1,7,12389.7,7798.0,2678.5,819.7,good
    1,2,106.0,106.0,4010.4,0.05,spam
    ```
    """)


# Main App Logic
CSV_FILE = 'data/url_stream.csv'
df = load_data(CSV_FILE)

# Apply data transformations if data exists
if df is not None:
    # Apply domain extraction globally
    if 'domain' not in df.columns:
        df['domain'] = df['url'].apply(get_domain)
    
    # Enrich with user data (handles, profile URLs, follower ratios) once at the start
    if 'handle' not in df.columns:
        with st.spinner('Fetching user data...'):
            df = enrich_with_user_data(df)

# Always show navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Suspicious Authors", "Run Classifier", "General Stats"])

# Handle page routing
if page == "Run Classifier":
    # Run Classifier doesn't need url_stream.csv data
    show_classifier_page()
elif df is None:
    st.error(f"File `{CSV_FILE}` not found. Please run the firehose script first to collect data.")
    st.info("You can still use the **Run Classifier** page to test uploaded CSV files.")
elif page == "Suspicious Authors":
    show_suspicious_authors(df)
elif page == "General Stats":
    show_general_stats(df)
