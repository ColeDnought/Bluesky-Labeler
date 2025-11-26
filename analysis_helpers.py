"""
URL Analysis Transformations

This module provides functions for analyzing URL posting patterns in social media data.
"""

import pandas as pd
from urllib.parse import urlparse
import asyncio
from fetch_users import get_follower_ratios


def load_url_data(csv_file='url_stream.csv'):
    """
    Load URL data from CSV file with proper timestamp parsing.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and cleaned dataframe
    """
    df = pd.read_csv(csv_file, parse_dates=['timestamp'], low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="ISO8601", utc=True)
    df = df.dropna(subset=['timestamp'])
    return df


def get_domain(url):
    """
    Extract domain from URL.
    
    Args:
        url (str): URL to parse
        
    Returns:
        str: Domain name without 'www.' prefix, or 'invalid'/'error' if parsing fails
    """
    try:
        if not isinstance(url, str):
            return 'invalid'
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return 'error'


def add_domain_column(df):
    """
    Add domain column to dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with 'url' column
        
    Returns:
        pd.DataFrame: Dataframe with added 'domain' column
    """
    df = df.copy()
    df['domain'] = df['url'].apply(get_domain)
    return df


def analyze_suspicious_authors(df, min_posts=5, domain_share_threshold=0.6):
    """
    Identify suspicious authors who post the same domain repeatedly.
    
    Args:
        df (pd.DataFrame): Dataframe with 'author', 'url', 'domain', and 'timestamp' columns
        min_posts (int): Minimum number of posts required
        domain_share_threshold (float): Minimum fraction of posts to same domain (0-1)
        
    Returns:
        pd.DataFrame: Suspicious authors with statistics including:
            - author: Author identifier
            - total_posts: Total number of posts
            - unique_domains: Number of unique domains posted
            - domain: Most frequently posted domain
            - domain_count: Number of posts to top domain
            - domain_share: Fraction of posts to top domain
            - min/max: First and last post timestamps
            - duration: Time span of posts
            - duration_seconds: Duration in seconds
            - posts_per_minute: Posting frequency
    """
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
        (author_analysis['total_posts'] >= min_posts) & 
        (author_analysis['domain_share'] >= domain_share_threshold)
    ].copy()

    # 5. Calculate time period and frequency
    suspicious_posts = df[df['author'].isin(suspicious_authors['author'])]

    # Group by author to get time range
    time_stats = suspicious_posts.groupby('author')['timestamp'].agg(['min', 'max']).reset_index()
    time_stats['duration'] = time_stats['max'] - time_stats['min']
    time_stats['duration_seconds'] = time_stats['duration'].apply(lambda x: x.total_seconds())

    # Merge time stats back
    suspicious_authors = pd.merge(suspicious_authors, time_stats, on='author')

    # Calculate frequency (posts per minute)
    # Add a small epsilon to duration to avoid division by zero
    suspicious_authors['posts_per_minute'] = suspicious_authors['total_posts'] / (
        (suspicious_authors['duration_seconds'] / 60) + 0.001
    )

    # Sort by frequency
    suspicious_authors = suspicious_authors.sort_values('posts_per_minute', ascending=False)
    
    return suspicious_authors


def analyze_url_bursts(df, min_url_count=5, min_gap_threshold=10):
    """
    Analyze URL posting patterns to detect burst activity.
    
    Args:
        df (pd.DataFrame): Dataframe with 'url' and 'timestamp' columns
        min_url_count (int): Minimum number of posts for a URL to be considered
        min_gap_threshold (float): Threshold in seconds for identifying bursty URLs
        
    Returns:
        tuple: (url_stats, bursty_urls)
            - url_stats (pd.DataFrame): All frequent URLs with burst statistics
            - bursty_urls (pd.DataFrame): URLs with potential burst activity (min_gap < threshold)
    """
    # 1. Count posts per URL
    url_counts = df['url'].value_counts().reset_index()
    url_counts.columns = ['url', 'count']

    # 2. Filter for frequent URLs
    frequent_urls = url_counts[url_counts['count'] >= min_url_count]['url'].tolist()
    burst_df = df[df['url'].isin(frequent_urls)].copy()

    # 3. Calculate time gaps
    burst_df = burst_df.sort_values(['url', 'timestamp'])
    burst_df['prev_timestamp'] = burst_df.groupby('url')['timestamp'].shift(1)
    burst_df['time_gap'] = (burst_df['timestamp'] - burst_df['prev_timestamp']).dt.total_seconds()

    # 4. Aggregate stats per URL
    url_stats = burst_df.groupby('url').agg(
        count=('timestamp', 'count'),
        min_gap=('time_gap', 'min'),
        avg_gap=('time_gap', 'mean'),
        std_gap=('time_gap', 'std')
    ).reset_index()

    # 5. Sort by count descending, then by avg_gap ascending
    url_stats = url_stats.sort_values(['count', 'avg_gap'], ascending=[False, True])

    # 6. Identify "Burst" URLs
    bursty_urls = url_stats[url_stats['min_gap'] < min_gap_threshold].sort_values('avg_gap')
    
    return url_stats, bursty_urls


def analyze_authors_comprehensive(df, labels_df=None):
    """
    Create a comprehensive per-author dataframe with all available statistics.
    
    Args:
        df (pd.DataFrame): Dataframe with 'author', 'url', 'domain', and 'timestamp' columns
        labels_df (pd.DataFrame, optional): Dataframe with 'author' and 'label' columns to merge in
        
    Returns:
        pd.DataFrame: Comprehensive author statistics including:
            - author: Author identifier
            - label: Author label (if labels_df provided)
            - total_posts: Total number of posts
            - unique_domains: Number of unique domains posted
            - unique_urls: Number of unique URLs posted
            - top_domain: Most frequently posted domain
            - top_domain_count: Number of posts to top domain
            - domain_share: Fraction of posts to top domain
            - first_post: Timestamp of first post
            - last_post: Timestamp of last post
            - duration: Time span of posts
            - duration_seconds: Duration in seconds
            - posts_per_minute: Posting frequency
            - avg_time_between_posts: Average time between consecutive posts (seconds)
            - posts_to_bursty_urls: Number of posts to URLs with burst patterns
    """
    # 1. Basic author stats
    author_stats = df.groupby('author').agg(
        total_posts=('url', 'count'),
        unique_domains=('domain', 'nunique'),
        unique_urls=('url', 'nunique'),
        first_post=('timestamp', 'min'),
        last_post=('timestamp', 'max')
    ).reset_index()
    
    # 2. Top domain per author
    top_domains = df.groupby(['author', 'domain']).size().reset_index(name='top_domain_count')
    top_domains = top_domains.sort_values(['author', 'top_domain_count'], ascending=[True, False])
    top_domains = top_domains.groupby('author').first().reset_index()
    top_domains = top_domains.rename(columns={'domain': 'top_domain'})
    
    # 3. Merge basic stats with top domain
    author_analysis = pd.merge(author_stats, top_domains, on='author')
    author_analysis['domain_share'] = author_analysis['top_domain_count'] / author_analysis['total_posts']
    
    # 4. Calculate time-based metrics
    author_analysis['duration'] = author_analysis['last_post'] - author_analysis['first_post']
    author_analysis['duration_seconds'] = author_analysis['duration'].apply(lambda x: x.total_seconds())
    
    # Posts per minute (add epsilon to avoid division by zero)
    author_analysis['posts_per_minute'] = author_analysis['total_posts'] / (
        (author_analysis['duration_seconds'] / 60) + 0.001
    )
    
    # 5. Calculate average time between posts (optimized version)
    df_sorted = df.sort_values(['author', 'timestamp'])
    df_sorted['time_gap'] = df_sorted.groupby('author')['timestamp'].diff().dt.total_seconds()
    avg_gaps = df_sorted.groupby('author')['time_gap'].mean().reset_index()
    avg_gaps.columns = ['author', 'avg_time_between_posts']
    author_analysis = pd.merge(author_analysis, avg_gaps, on='author', how='left')
    
    # 6. Count posts to bursty URLs
    # First identify bursty URLs (URLs posted >= 5 times with min gap < 10 seconds)
    url_counts = df['url'].value_counts()
    frequent_urls = url_counts[url_counts >= 5].index.tolist()
    
    if len(frequent_urls) > 0:
        burst_df = df[df['url'].isin(frequent_urls)].copy()
        burst_df = burst_df.sort_values(['url', 'timestamp'])
        burst_df['time_gap'] = burst_df.groupby('url')['timestamp'].diff().dt.total_seconds()
        
        bursty_url_stats = burst_df.groupby('url')['time_gap'].min().reset_index()
        bursty_urls = bursty_url_stats[bursty_url_stats['time_gap'] < 10]['url'].tolist()
        
        bursty_posts = df[df['url'].isin(bursty_urls)].groupby('author').size().reset_index(name='posts_to_bursty_urls')
        author_analysis = pd.merge(author_analysis, bursty_posts, on='author', how='left')
        author_analysis['posts_to_bursty_urls'] = author_analysis['posts_to_bursty_urls'].fillna(0).astype(int)
    else:
        author_analysis['posts_to_bursty_urls'] = 0
    
    # 7. Merge in labels if provided
    if labels_df is not None and 'label' in labels_df.columns:
        author_analysis = pd.merge(
            labels_df[['author', 'label']], 
            author_analysis, 
            on='author', 
            how='right'
        )
    
    # Sort by total posts descending
    author_analysis = author_analysis.sort_values('total_posts', ascending=False)
    
    return author_analysis

def populate_follower_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds follower/following counts and ratio to dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with 'author' column containing DIDs
        
    Returns:
        pd.DataFrame: Dataframe with added columns:
            - followers_count: Number of followers
            - follows_count: Number of accounts followed
            - follower_following_ratio: Ratio of followers to following
    """
    unique_authors = df['author'].dropna().unique()

    if len(unique_authors) == 0:
        # Ensure the expected columns exist even when there is nothing to fetch
        df = df.copy()
        df['followers_count'] = pd.NA
        df['follows_count'] = pd.NA
        df['follower_following_ratio'] = pd.NA
        return df

    follower_data = asyncio.run(get_follower_ratios(list(unique_authors)))

    # Merge follower data into the dataframe
    df_enriched = df.merge(
        follower_data[['did', 'followers_count', 'follows_count', 'follower_following_ratio']],
        left_on='author',
        right_on='did',
        how='left'
    )

    # Drop the duplicate 'did' column from the merge
    if 'did' in df_enriched.columns and 'did' not in df.columns:
        df_enriched = df_enriched.drop(columns=['did'])

    return df_enriched


def full_analysis_pipeline(csv_file='url_stream.csv', 
                          min_posts=5,
                          domain_share_threshold=0.6,
                          min_url_count=5,
                          min_gap_threshold=10):
    """
    Run complete URL analysis pipeline.
    
    Args:
        csv_file (str): Path to the CSV file
        min_posts (int): Minimum posts for suspicious author detection
        domain_share_threshold (float): Domain share threshold for suspicious authors
        min_url_count (int): Minimum URL count for burst analysis
        min_gap_threshold (float): Time gap threshold (seconds) for burst detection
        
    Returns:
        dict: Dictionary containing:
            - 'df': Original dataframe with domain column added
            - 'suspicious_authors': Suspicious authors analysis
            - 'url_stats': URL burst statistics
            - 'bursty_urls': URLs with burst activity
            - 'author_stats': Comprehensive per-author statistics
    """
    # Load data
    df = load_url_data(csv_file)
    
    # Add domain column
    df = add_domain_column(df)
    
    # Analyze suspicious authors
    suspicious_authors = analyze_suspicious_authors(
        df, 
        min_posts=min_posts, 
        domain_share_threshold=domain_share_threshold
    )
    
    # Analyze URL bursts
    url_stats, bursty_urls = analyze_url_bursts(
        df, 
        min_url_count=min_url_count,
        min_gap_threshold=min_gap_threshold
    )
    
    # Comprehensive author analysis
    author_stats = analyze_authors_comprehensive(df)
    
    return {
        'df': df,
        'suspicious_authors': suspicious_authors,
        'url_stats': url_stats,
        'bursty_urls': bursty_urls,
        'author_stats': author_stats
    }


def augment_data(
        dataframe: pd.DataFrame, 
        feature_columns: list[str], 
        target_column: str, 
        num_synthetic_rows: int,
    ) -> pd.DataFrame:
    """
    Generate synthetic data to augment training set.
    
    Args:
        dataframe: Original data with features and target
        feature_columns: List of feature column names
        target_column: Name of the target/label column
        num_synthetic_rows: Number of synthetic rows to generate
        per_class: If True, generate num_synthetic_rows PER CLASS to preserve
                   class distributions. If False, generate total rows.
    
    Returns:
        DataFrame with synthetic data (does not include original data)
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    
    original_data = dataframe[feature_columns + [target_column]].copy()
    
    # Generate synthetic data separately for each class to preserve distributions
    synthetic_parts = []
    
    for label in original_data[target_column].unique():
        class_data = original_data[original_data[target_column] == label].copy()
        
        if len(class_data) < 3:
            # Not enough data to synthesize, just duplicate
            synthetic_parts.append(class_data.sample(n=num_synthetic_rows, replace=True))
            continue
        
        metadata = Metadata.detect_from_dataframe(class_data)
        metadata.update_column(column_name=target_column, sdtype='categorical')
        
        # Use CTGAN for better handling of non-Gaussian distributions
        synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=False)
        synthesizer.fit(class_data)
        synthetic_class = synthesizer.sample(num_rows=num_synthetic_rows)
        
        synthetic_parts.append(synthetic_class)
    
    return pd.concat(synthetic_parts, ignore_index=True)
