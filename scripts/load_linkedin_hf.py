"""
LinkedIn Data Loader from HuggingFace

This script downloads the LinkedIn data from HuggingFace dataset/space.
Used for production deployment where LinkedIn data is stored on HuggingFace.

Environment Variables:
    HF_LINKEDIN_REPO: HuggingFace repo ID (e.g., "username/career-agent-data")
    HF_LINKEDIN_FILE: Filename in the repo (default: "linkedin.txt")
    HF_TOKEN: HuggingFace token for private repos (optional)

Usage:
    python scripts/load_linkedin_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configuration
ME_DIR = Path("me")
OUTPUT_FILE = ME_DIR / "linkedin.txt"

# HuggingFace settings
HF_LINKEDIN_REPO = os.getenv("HF_LINKEDIN_REPO", "")
HF_LINKEDIN_FILE = os.getenv("HF_LINKEDIN_FILE", "linkedin.txt")
HF_TOKEN = os.getenv("HF_TOKEN", None)


def download_from_huggingface():
    """Download LinkedIn data from HuggingFace."""
    
    if not HF_LINKEDIN_REPO:
        print("❌ HF_LINKEDIN_REPO environment variable not set")
        print("   Set it to your HuggingFace repo ID, e.g., 'username/career-agent-data'")
        return None
    
    print(f"📥 Downloading LinkedIn data from HuggingFace...")
    print(f"   Repo: {HF_LINKEDIN_REPO}")
    print(f"   File: {HF_LINKEDIN_FILE}")
    
    try:
        # Ensure output directory exists
        ME_DIR.mkdir(exist_ok=True)
        
        # Download file
        local_path = hf_hub_download(
            repo_id=HF_LINKEDIN_REPO,
            filename=HF_LINKEDIN_FILE,
            repo_type="dataset",  # Use "space" if storing in a Space
            token=HF_TOKEN,
            local_dir=ME_DIR,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Downloaded to: {local_path}")
        
        # Read and return content
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   Content length: {len(content):,} characters")
        return content
        
    except Exception as e:
        print(f"❌ Failed to download from HuggingFace: {e}")
        return None


def load_linkedin_data():
    """
    Load LinkedIn data, trying local file first, then HuggingFace.
    
    Returns:
        str: LinkedIn profile text, or empty string if not found
    """
    
    # First, try local file
    if OUTPUT_FILE.exists():
        print(f"📄 Loading LinkedIn data from local file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.strip():
            print(f"   Content length: {len(content):,} characters")
            return content
    
    # Try HuggingFace
    if HF_LINKEDIN_REPO:
        content = download_from_huggingface()
        if content:
            return content
    
    # Fallback: check for PDF (legacy support)
    pdf_file = ME_DIR / "linkedin.pdf"
    if pdf_file.exists():
        print(f"📄 Found legacy PDF file: {pdf_file}")
        print("   Run 'python extract_linkedin.py' to extract text")
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except ImportError:
            print("   pypdf not installed, cannot read PDF")
    
    print("⚠️ No LinkedIn data found")
    print("   Options:")
    print("   1. Place your LinkedIn ZIP export in me/ and run:")
    print("      python scripts/extract_linkedin_zip.py")
    print("   2. Set HF_LINKEDIN_REPO to download from HuggingFace")
    
    return ""


if __name__ == "__main__":
    content = load_linkedin_data()
    if content:
        print("\n--- Preview (first 500 chars) ---")
        print(content[:500])
