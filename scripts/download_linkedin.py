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
from playwright.async_api import async_playwright, BrowserContext

# Configuration
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")
LINKEDIN_PROFILE_URL = "https://www.linkedin.com/in/mayank-malhotra-858917217/"
LINKEDIN_SKIP_LOGIN = os.getenv("LINKEDIN_SKIP_LOGIN", "false").strip().lower() in {"1", "true", "yes"}
OUTPUT_DIR = Path("me")
OUTPUT_FILE = OUTPUT_DIR / "linkedin.pdf"

async def stealth_context(context: BrowserContext):
    """Apply stealth techniques to bypass bot detection."""
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
        
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        window.chrome = {
            runtime: {},
        };
        
        Object.defineProperty(navigator, 'permissions', {
            get: () => ({
                query: () => Promise.resolve({ state: 'prompt' })
            }),
        });
    """)

async def download_linkedin_profile():
    """Download LinkedIn profile as PDF using Playwright."""
    
    if not LINKEDIN_SKIP_LOGIN:
        if not LINKEDIN_EMAIL or not LINKEDIN_PASSWORD:
            print("❌ Error: LINKEDIN_EMAIL and LINKEDIN_PASSWORD environment variables must be set")
            print("   For GitHub Actions: Add them as repository secrets")
            print("   For local use: Set them in your environment or .env file")
            print("   Or set LINKEDIN_SKIP_LOGIN=true to attempt public profile download without credentials")
            sys.exit(1)
    if not LINKEDIN_PROFILE_URL:
        print("❌ Error: LINKEDIN_PROFILE_URL environment variable must be set")
        print("   Example: https://www.linkedin.com/in/your-handle/")
        print("   For GitHub Actions: Add it as a repository secret or env")
        sys.exit(1)
    
    print("🚀 Starting LinkedIn profile download...")
    
    async with async_playwright() as p:
        # Launch browser
        print("📱 Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-popup-blocking',
                '--disable-extensions',
                '--disable-sync',
                '--disable-web-resources',
                '--disable-default-apps',
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            geolocation={'latitude': 40.7128, 'longitude': -74.0060},
            permissions=['geolocation'],
        )
        
        # Apply stealth techniques
        await stealth_context(context)
        
        page = await context.new_page()
        
        # Set additional headers to appear human-like
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        try:
            if not LINKEDIN_SKIP_LOGIN:
                # Navigate to LinkedIn login
                print("🔑 Logging into LinkedIn...")
                await page.goto('https://www.linkedin.com/login', wait_until='domcontentloaded', timeout=60000)
                
                # Add random delays to appear human-like
                await asyncio.sleep(2)
                
                # Fill login form with delay between keystrokes
                username_field = await page.query_selector('input[id="username"]')
                if username_field:
                    await username_field.click()
                    await asyncio.sleep(0.5)
                    for char in LINKEDIN_EMAIL:
                        await page.type('input[id="username"]', char, delay=50)
                        await asyncio.sleep(0.05)
                else:
                    await page.fill('input[id="username"]', LINKEDIN_EMAIL)
                
                await asyncio.sleep(1)
                
                password_field = await page.query_selector('input[id="password"]')
                if password_field:
                    await password_field.click()
                    await asyncio.sleep(0.5)
                    for char in LINKEDIN_PASSWORD:
                        await page.type('input[id="password"]', char, delay=50)
                        await asyncio.sleep(0.05)
                else:
                    await page.fill('input[id="password"]', LINKEDIN_PASSWORD)
                
                await asyncio.sleep(1.5)
                
                # Click login button
                await page.click('button[type="submit"]')
                
                # Wait for navigation with longer timeout
                await page.wait_for_load_state('domcontentloaded', timeout=90000)
                await asyncio.sleep(3)
                
                # Check if login was successful
                current_url = page.url
                if 'checkpoint' in current_url or 'challenge' in current_url:
                    print("⚠️  Warning: LinkedIn security challenge detected")
                    print("   You may need to verify your login manually")
                    # Take screenshot for debugging
                    await page.screenshot(path='linkedin_challenge.png')
                    print("   Screenshot saved as linkedin_challenge.png")
                    print("❌ Exiting due to login challenge. Configure trusted device, reduce 2FA prompts, or reuse cookies.")
                    sys.exit(2)
            else:
                print("🔓 Skipping login (LINKEDIN_SKIP_LOGIN=true). Attempting public profile view...")
            
            # Navigate to profile (assuming we're logged in)
            print("📄 Navigating to profile...")
            await asyncio.sleep(2)
            await page.goto(LINKEDIN_PROFILE_URL, wait_until='domcontentloaded', timeout=90000)
            # Detect redirects to login/authwall when skipping login
            if LINKEDIN_SKIP_LOGIN:
                current_url = page.url
                if any(x in current_url for x in ["/login", "checkpoint", "challenge", "authwall"]):
                    print("❌ Public profile not accessible without login. Enable 'Public profile visibility' in LinkedIn settings or disable LINKEDIN_SKIP_LOGIN.")
                    # Screenshot for debugging
                    await page.screenshot(path='linkedin_public_blocked.png')
                    print("   Screenshot saved as linkedin_public_blocked.png")
                    sys.exit(3)
            
            # Wait for profile content to load
            await page.wait_for_selector('main', timeout=20000)
            
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
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate PDF
            print(f"💾 Generating PDF to {OUTPUT_FILE}...")
            # Print the current page to PDF using Chromium
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
