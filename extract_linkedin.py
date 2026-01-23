"""
Script to extract text from LinkedIn PDF for use in the static site.
Run this script to generate linkedin.txt from linkedin.pdf
"""
from pypdf import PdfReader

def extract_linkedin_text():
    try:
        reader = PdfReader("me/linkedin.pdf")
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        with open("me/linkedin.txt", "w", encoding="utf-8") as f:
            f.write(text)
        
        print("✓ Successfully extracted LinkedIn profile text to me/linkedin.txt")
        print(f"  Extracted {len(text)} characters from {len(reader.pages)} pages")
        
    except FileNotFoundError:
        print("✗ Error: me/linkedin.pdf not found")
        print("  Please ensure your LinkedIn PDF is saved as me/linkedin.pdf")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    extract_linkedin_text()
