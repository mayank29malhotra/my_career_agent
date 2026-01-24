"""
Automated LinkedIn Profile PDF Download Script

This script logs into LinkedIn and downloads your profile as a PDF.
It's designed to run in GitHub Actions but can also be run locally.

Requirements:
- playwright
- Environment variables: LINKEDIN_EMAIL, LINKEDIN_PASSWORD

Usage:
    python scripts/download_linkedin.py
"""

import os
import sys
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
OUTPUT_DIR = Path("me")
OUTPUT_FILE = OUTPUT_DIR / "linkedin.pdf"

async def download_linkedin_profile():
    """Download LinkedIn profile as PDF using Playwright."""
    
    if not LINKEDIN_EMAIL or not LINKEDIN_PASSWORD:
        print("❌ Error: LINKEDIN_EMAIL and LINKEDIN_PASSWORD environment variables must be set")
        print("   For GitHub Actions: Add them as repository secrets")
        print("   For local use: Set them in your environment or .env file")
        sys.exit(1)
    
    print("🚀 Starting LinkedIn profile download...")
    
    async with async_playwright() as p:
        # Launch browser
        print("📱 Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        
        try:
            # Navigate to LinkedIn login with increased timeout
            print("🔑 Logging into LinkedIn...")
            await page.goto('https://www.linkedin.com/login', wait_until='load', timeout=60000)
            
            # Fill login form
            await page.fill('input[id="username"]', LINKEDIN_EMAIL)
            await page.fill('input[id="password"]', LINKEDIN_PASSWORD)
            
            # Click login button
            await page.click('button[type="submit"]')
            
            # Wait for navigation
            await page.wait_for_load_state('networkidle', timeout=30000)
            
            # Check if login was successful
            current_url = page.url
            if 'checkpoint' in current_url or 'challenge' in current_url:
                print("⚠️  Warning: LinkedIn security challenge detected")
                print("   You may need to verify your login manually")
                # Take screenshot for debugging
                await page.screenshot(path='linkedin_challenge.png')
                print("   Screenshot saved as linkedin_challenge.png")
            
            # Navigate to profile (assuming we're logged in)
            print("📄 Navigating to profile...")
            await page.goto('https://www.linkedin.com/in/me/', wait_until='networkidle', timeout=30000)
            
            # Wait for profile content to load
            await page.wait_for_selector('main', timeout=10000)
            
            # Scroll to load all content
            print("📜 Loading full profile...")
            await page.evaluate("""
                async () => {
                    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
                    for (let i = 0; i < 5; i++) {
                        window.scrollTo(0, document.body.scrollHeight);
                        await delay(1000);
                    }
                    window.scrollTo(0, 0);
                }
            """)
            
            # Wait a bit for images to load
            await asyncio.sleep(2)
            
            # Ensure output directory exists
            OUTPUT_DIR.mkdir(exist_ok=True)
            
            # Generate PDF
            print(f"💾 Generating PDF to {OUTPUT_FILE}...")
            await page.pdf(
                path=str(OUTPUT_FILE),
                format='A4',
                print_background=True,
                margin={
                    'top': '0.5in',
                    'right': '0.5in',
                    'bottom': '0.5in',
                    'left': '0.5in'
                }
            )
            
            print(f"✅ Successfully downloaded LinkedIn profile to {OUTPUT_FILE}")
            print(f"   File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"❌ Error during download: {e}")
            # Take screenshot for debugging
            try:
                await page.screenshot(path='linkedin_error.png')
                print("   Debug screenshot saved as linkedin_error.png")
            except:
                pass
            sys.exit(1)
            
        finally:
            await browser.close()

def main():
    """Main entry point."""
    try:
        asyncio.run(download_linkedin_profile())
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
