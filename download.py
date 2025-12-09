#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import requests
import tarfile
import os
import sys


def create_data_directory():
    """Create 'data' directory if it does not exist."""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def read_text_file(file_path):
    """Read the tab-separated text file into a DataFrame."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def download_and_decompress(df):
    """Download, decompress, and clean up each file into 'data' directory."""
    data_dir = 'data'
    for idx, row in df.iterrows():
        file_name = row['file_name']
        cdn_link = row['cdn_link']
        print(f"Downloading {file_name}...")
        resp = requests.get(cdn_link, stream=True)
        if resp.status_code == 200:
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {file_name}")
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(data_dir)
                print(f"Decompressed {file_name}")
                os.remove(file_path)
                print(f"Deleted {file_name}")
            except Exception as e:
                print(f"Error decompressing {file_name}: {e}")
        else:
            print(f"Failed to download {file_name} (status code: {resp.status_code})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python download.py <path_to_your_text_file>")
        sys.exit(1)
    create_data_directory()
    file_path = sys.argv[1]
    df = read_text_file(file_path)
    if df is not None:
        download_and_decompress(df)


if __name__ == "__main__":
    main()
