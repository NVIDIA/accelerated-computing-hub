#!/usr/bin/env python3

import urllib.request
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not DATA_DIR.exists():
        print(f"Creating data directory at {DATA_DIR}...")
        DATA_DIR.mkdir()


def download_file(url, filename, description):
    """Download a file if it doesn't exist"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Downloading {description}...")
        urllib.request.urlretrieve(url, str(filepath))
    else:
        print(f"{filename} already exists, skipping download")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for RAPIDS tutorial")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    
    parser.add_argument(
        "--nyc-parking",
        action="store_true",
        help="Download NYC parking violations dataset",
    )
    
    args = parser.parse_args()

    # If no specific flags are provided, show help and exit
    if not any(vars(args).values()):
        parser.print_help()
        return

    print("Checking and downloading datasets...")
    create_data_directory()


    if args.all or args.nyc_parking:
        download_file(
            "https://data.rapids.ai/datasets/nyc_parking/nyc_parking_violations_2022.parquet",
            "nyc_parking_violations_2022.parquet",
            "NYC parking violations dataset",
        )

    print("Download complete!")

if __name__ == "__main__":
    main() 
