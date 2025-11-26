import asyncio
import atproto
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Load environment variables
load_dotenv()
USERNAME = os.getenv("BSKY_USR")
PASSWORD = os.getenv("BSKY_PWD")


async def fetch_single_user(client, did):
    """Fetch info for a single DID asynchronously."""
    try:
        # Resolve the DID to get the profile
        profile = await client.com.atproto.repo.describe_repo({'repo': did})
        handle = profile.handle
        profile_url = f"https://bsky.app/profile/{handle}"
        
        return {
            'did': did,
            'handle': handle,
            'profile_url': profile_url
        }
    except Exception:
        # If we can't resolve, keep the DID
        return {
            'did': did,
            'handle': did,
            'profile_url': f"https://bsky.app/profile/{did}"
        }


async def fetch_user_with_ratio(client, did):
    """Fetch follower:following ratio for a single DID asynchronously."""
    try:
        # Get the full profile with follower/following counts
        profile = await client.app.bsky.actor.get_profile({'actor': did})
        
        followers_count = profile.followers_count or 1
        follows_count = profile.follows_count or 1
        
        return {
            'did': did,
            'handle': profile.handle,
            'profile_url': f"https://bsky.app/profile/{profile.handle}",
            'followers_count': followers_count,
            'follows_count': follows_count,
            'follower_following_ratio': followers_count / follows_count
        }
    except Exception:
        return {
            'did': did,
            'handle': did,
            'profile_url': f"https://bsky.app/profile/{did}",
            'followers_count': None,
            'follows_count': None,
            'follower_following_ratio': None
        }
    
async def add_follower_column(dids: list[str]):
    """Fetch follower:following ratio"""
    pass

async def get_user_info(dids):
    """
    Fetch handles and profile links for a list of Bluesky DIDs asynchronously.
    
    Args:
        dids: List of DID identifiers (e.g., ['did:plc:uld74vzf773y7ovqqm2jfaft'])
    
    Returns:
        DataFrame with columns: did, handle, profile_url
    """
    client = atproto.AsyncClient()
    
    # Fetch all DIDs concurrently
    tasks = [fetch_single_user(client, did) for did in dids]
    results = await tqdm_asyncio.gather(*tasks)
    
    return pd.DataFrame(results)


async def get_follower_ratios(dids):
    """
    Fetch follower:following ratios for a list of Bluesky DIDs asynchronously.
    
    Args:
        dids: List of DID identifiers (e.g., ['did:plc:uld74vzf773y7ovqqm2jfaft'])
    
    Returns:
        DataFrame with columns: did, handle, followers_count, follows_count, follower_following_ratio
    """
    client = atproto.AsyncClient()
    
    # Login to authenticate
    if USERNAME and PASSWORD:
        await client.login(USERNAME, PASSWORD)
    
    # Fetch all DIDs concurrently
    tasks = [fetch_user_with_ratio(client, did) for did in dids]
    results = await asyncio.gather(*tasks)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    sample_dids = [
        "did:plc:uld74vzf773y7ovqqm2jfaft",
        "did:plc:hacoy4ddxz2wyeagydupyzzo"
    ]
    
    print("Fetching basic user info...")
    df = asyncio.run(get_user_info(sample_dids))
    print(df)
    
    print("\n" + "="*60)
    print("Fetching follower:following ratios...")
    print("="*60)
    ratios_df = asyncio.run(get_follower_ratios(sample_dids))
    
    # Set pandas display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(ratios_df.to_string(index=False))