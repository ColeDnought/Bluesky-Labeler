import asyncio
import atproto
import pandas as pd


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
    except Exception as e:
        # If we can't resolve, keep the DID
        return {
            'did': did,
            'handle': did,
            'profile_url': f"https://bsky.app/profile/{did}"
        }


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
    results = await asyncio.gather(*tasks)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    sample_dids = [
        "did:plc:uld74vzf773y7ovqqm2jfaft",
        "did:plc:hacoy4ddxz2wyeagydupyzzo"
    ]
    
    df = asyncio.run(get_user_info(sample_dids))
    print(df)