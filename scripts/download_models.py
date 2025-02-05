#!/usr/bin/env python3

import os
import json
import requests
from tqdm import tqdm
import urllib.parse

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def download_file(url, destination_path):
    """
    Download a file with progress bar.
    Returns True if successful, False otherwise.
    """
    try:
        # Get the filename from the URL
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        if '?' in filename:  # Remove query parameters from filename
            filename = filename.split('?')[0]
        
        # If filename is empty or doesn't have an extension, try to get it from content-disposition
        if not filename or '.' not in filename:
            response = requests.get(url, stream=True, allow_redirects=True)
            if 'content-disposition' in response.headers:
                content_disposition = response.headers['content-disposition']
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')

        full_path = os.path.join(destination_path, filename)
        
        # Skip if file already exists
        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return True

        # Make the request
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))

        # Open the local file to write the downloaded content
        with open(full_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

if __name__ == "__main__":
    # Base directory
    base_dir = "ComfyUI/models"

    # Dictionary of directories and their corresponding URLs
    with open("custom_models.json", "r") as file:
        downloads = json.load(file) 

    # Process each directory and its downloads
    for directory, urls in downloads.items():
        full_path = os.path.join(base_dir, directory)
        ensure_directory(full_path)
        
        for url in urls:
            print(f"\nDownloading to {directory}:")
            success = download_file(url, full_path)
            if success:
                print(f"Successfully downloaded to {directory}")
            else:
                print(f"Failed to download {url}")