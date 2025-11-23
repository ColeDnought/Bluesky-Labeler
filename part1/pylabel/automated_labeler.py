"""Implementation of automated moderator"""

import csv
import os
from typing import List
import requests
from PIL import Image
import imagehash
from io import BytesIO

from atproto import Client
from .label import post_from_url

T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.27

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        self.ts_words = set()
        self.ts_domains = set()
        self.news_domains = {}
        self.dog_hashes = []

        # Load T&S words
        with open(os.path.join(input_dir, "t-and-s-words.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Word"]:
                    self.ts_words.add(row["Word"].lower())

        # Load T&S domains
        with open(os.path.join(input_dir, "t-and-s-domains.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Domain"]:
                    self.ts_domains.add(row["Domain"].lower())

        # Load News domains
        with open(os.path.join(input_dir, "news-domains.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Domain"] and row["Source"]:
                    self.news_domains[row["Domain"].lower()] = row["Source"]

        # Load Dog images and compute hashes
        dog_dir = os.path.join(input_dir, "dog-list-images")
        if os.path.exists(dog_dir):
            for img_name in os.listdir(dog_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dog_dir, img_name)
                    try:
                        with Image.open(img_path) as img:
                            self.dog_hashes.append(imagehash.phash(img))
                    except Exception as e:
                        print(f"Error loading dog image {img_name}: {e}")

    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        labels = set()
        try:
            post = post_from_url(self.client, url)
            post_text = post.value.text.lower()
            
            # Check T&S words
            for word in self.ts_words:
                if word in post_text:
                    labels.add(T_AND_S_LABEL)
                    break # Optimization: one match is enough for this label

            # Check Domains in Text (covering both facets and raw text)
            # We iterate over all domains and check if they are in the text.
            # This covers the case where facets are missing.
            for domain in self.ts_domains:
                if domain in post_text:
                    labels.add(T_AND_S_LABEL)
            
            for domain, source in self.news_domains.items():
                if domain in post_text:
                    labels.add(source)

            # Check Images
            if hasattr(post.value, 'embed') and post.value.embed:
                # Handle images embed
                images = []
                if hasattr(post.value.embed, 'images'):
                     images = post.value.embed.images
                elif hasattr(post.value.embed, 'media') and hasattr(post.value.embed.media, 'images'):
                     # Handle recordWithMedia
                     images = post.value.embed.media.images
                
                for image in images:
                    if hasattr(image, 'image'):
                        # Construct blob URL
                        did = post.uri.split('/')[2]
                        cid = image.image.ref.link
                        blob_url = f"https://bsky.social/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}"
                        
                        try:
                            resp = requests.get(blob_url, timeout=10)
                            if resp.status_code == 200:
                                img = Image.open(BytesIO(resp.content))
                                img_hash = imagehash.phash(img)
                                
                                for dog_hash in self.dog_hashes:
                                    distance = img_hash - dog_hash
                                    # Normalized distance for 64-bit hash
                                    normalized_dist = distance / 64.0
                                    
                                    if normalized_dist <= THRESH:
                                         labels.add(DOG_LABEL)
                                         # print(f"Matched dog image with distance {normalized_dist} (Threshold: {THRESH}) for post {url}")
                                         break
                        except Exception as e:
                            print(f"Error processing image in post {url}: {e}")

        except Exception as e:
            print(f"Error fetching or processing post {url}: {e}")
            
        return list(labels)