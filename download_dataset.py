"""
Dataset Download Helper Script
Attempts direct download from remote source with automatic robust fallback.
"""
from prepare_dataset import prepare_dataset

if __name__ == '__main__':
    print("[*] Initiating Student Dataset acquisition...")
    prepare_dataset(force_download=True)
