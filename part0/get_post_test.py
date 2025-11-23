"""Simple test to ensure that you can retreive posts"""

import os

from atproto import Client
from dotenv import load_dotenv

from pylabel import post_from_url

load_dotenv(override=True)
USERNAME = os.getenv("BSKY_USR")
PW = os.getenv("BSKY_PWD")


def main():
    """Main function"""
    client = Client()
    client.login(USERNAME, PW)
    result = post_from_url(
        client, "https://bsky.app/profile/labeler-test.bsky.social/post/3lksxxugg4k27"
    )
    print("Successfully loaded post:", result)
    print("Post text:", result.value.text)
    if hasattr(result.value, 'embed'):
        print("Embed:", result.value.embed)
    if hasattr(result.value, 'facets'):
        print("Facets:", result.value.facets)


if __name__ == "__main__":
    main()
