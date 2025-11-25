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
    page = st.sidebar.radio("Go to", ["General Stats", "Suspicious Authors"])

    if page == "General Stats":
        show_general_stats(df)
    elif page == "Suspicious Authors":
        show_suspicious_authors(df)
