import streamlit as st
import pandas as pd
from urllib.parse import urlparse

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

def show_general_stats(df):
    st.header("General Statistics")
    st.write(f"Total Records: {len(df)}")

    # Top Domains
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Most Common Domains")
        top_domains_series = df['domain'].value_counts().head(20)
        top_domains_df = pd.DataFrame({
            'domain': top_domains_series.index.astype(str),
            'count': top_domains_series.values
        })
        st.bar_chart(top_domains_df.set_index('domain')['count'])

    with col2:
        st.write("Top 20 Domains Count")
        st.dataframe(top_domains_df)

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
        (author_analysis['total_posts'] >= 5) & 
        (author_analysis['domain_share'] >= 0.6)
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

    st.subheader(f"Found {len(suspicious_authors)} Suspicious Authors")
    
    # Display table
    display_cols = ['author', 'domain', 'total_posts', 'domain_share', 'duration', 'posts_per_minute']
    st.dataframe(
        suspicious_authors[display_cols].style.format({
            'domain_share': '{:.1%}',
            'posts_per_minute': '{:.2f}'
        })
    )

    st.divider()
    st.subheader("Inspect Author Posts")
    
    selected_author = st.selectbox("Select an author to view posts:", suspicious_authors['author'])
    
    if selected_author:
        author_posts = df[df['author'] == selected_author].sort_values('timestamp', ascending=False)
        st.write(f"Posts by **{selected_author}** ({len(author_posts)} posts)")
        
        # Display posts with links
        author_posts['url_link'] = author_posts['url'].apply(ensure_scheme)
        
        st.dataframe(
            author_posts[['timestamp', 'url_link', 'text', 'domain']],
            column_config={
                'url_link': st.column_config.LinkColumn('URL'),
                'timestamp': st.column_config.DatetimeColumn('Time', format="D MMM YYYY, h:mm:ss a")
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

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["General Stats", "Suspicious Authors"])

    if page == "General Stats":
        show_general_stats(df)
    elif page == "Suspicious Authors":
        show_suspicious_authors(df)
