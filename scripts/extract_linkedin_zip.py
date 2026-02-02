"""
LinkedIn Data Export Extractor

This script extracts and processes your LinkedIn data export ZIP file.
LinkedIn provides a ZIP file containing multiple CSV files with your profile data.

Supported data files:
- Profile.csv - Basic profile info
- Positions.csv - Work experience
- Education.csv - Education history
- Skills.csv - Your skills
- Certifications.csv - Certifications
- Projects.csv - Projects
- Languages.csv - Languages you speak
- Recommendations_Received.csv - Recommendations

Usage:
    python scripts/extract_linkedin_zip.py <path_to_zip>
    python scripts/extract_linkedin_zip.py  # Uses default location in me/ folder
"""

import os
import sys
import zipfile
import csv
from pathlib import Path
from io import StringIO

# Default paths
ME_DIR = Path("me")
OUTPUT_FILE = ME_DIR / "linkedin.txt"
DATA_DIR = ME_DIR / "linkedin_data"


def find_linkedin_zip():
    """Find LinkedIn export ZIP file in me/ directory."""
    zip_patterns = [
        "Complete_LinkedInDataExport*.zip",
        "*LinkedIn*.zip",
        "*.zip"
    ]
    
    for pattern in zip_patterns:
        matches = list(ME_DIR.glob(pattern))
        if matches:
            # Return the most recent one
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


def read_csv_from_zip(zip_ref, filename):
    """Read a CSV file from the ZIP archive, checking multiple possible paths."""
    possible_paths = [
        filename,
        f"Complete_LinkedInDataExport*/{filename}",
    ]
    
    # Find the actual path in the zip
    actual_path = None
    for name in zip_ref.namelist():
        # Check if the file matches (accounting for folder prefixes)
        if name.endswith(f"/{filename}") or name == filename:
            actual_path = name
            break
    
    if not actual_path:
        return []
    
    try:
        with zip_ref.open(actual_path) as f:
            content = f.read().decode('utf-8')
            reader = csv.DictReader(StringIO(content))
            return list(reader)
    except KeyError:
        return []
    except Exception as e:
        print(f"  ⚠️ Could not read {filename}: {e}")
        return []


def extract_profile(zip_ref):
    """Extract profile information."""
    rows = read_csv_from_zip(zip_ref, "Profile.csv")
    if not rows:
        return ""
    
    profile = rows[0]
    lines = ["# PROFILE", ""]
    
    name_parts = [profile.get('First Name', ''), profile.get('Last Name', '')]
    name = ' '.join(p for p in name_parts if p)
    if name:
        lines.append(f"Name: {name}")
    
    if profile.get('Headline'):
        lines.append(f"Headline: {profile['Headline']}")
    
    if profile.get('Summary'):
        lines.append(f"\nSummary:\n{profile['Summary']}")
    
    if profile.get('Industry'):
        lines.append(f"\nIndustry: {profile['Industry']}")
    
    location_parts = [profile.get('Geo Location', ''), profile.get('Country', '')]
    location = ', '.join(p for p in location_parts if p)
    if location:
        lines.append(f"Location: {location}")
    
    return '\n'.join(lines) + '\n'


def extract_positions(zip_ref):
    """Extract work experience (newest first)."""
    rows = read_csv_from_zip(zip_ref, "Positions.csv")
    if not rows:
        return ""
    
    # Parse date for sorting (format: "Mon YYYY" or "YYYY")
    def parse_date(date_str):
        if not date_str:
            return 0
        months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        parts = date_str.lower().split()
        try:
            if len(parts) == 2:  # "Aug 2024"
                month = months.get(parts[0][:3], 1)
                year = int(parts[1])
                return year * 100 + month
            elif len(parts) == 1:  # "2024"
                return int(parts[0]) * 100
        except:
            pass
        return 0
    
    # Sort: current jobs first (no end date), then by start date descending
    def sort_key(pos):
        end = pos.get('Finished On', '').strip()
        start = pos.get('Started On', '')
        is_current = not end or end.lower() == 'present'
        start_val = parse_date(start)
        return (0 if is_current else 1, -start_val)
    
    rows = sorted(rows, key=sort_key)
    
    lines = ["\n# WORK EXPERIENCE (Most Recent First)", ""]
    
    for pos in rows:
        company = pos.get('Company Name', 'Unknown Company')
        title = pos.get('Title', 'Unknown Title')
        
        start = pos.get('Started On', '')
        end = pos.get('Finished On', 'Present')
        duration = f"{start} - {end}" if start else ""
        
        lines.append(f"## {title} at {company}")
        if duration:
            lines.append(f"Duration: {duration}")
        
        if pos.get('Location'):
            lines.append(f"Location: {pos['Location']}")
        
        if pos.get('Description'):
            lines.append(f"\n{pos['Description']}")
        
        lines.append("")
    
    return '\n'.join(lines)


def extract_education(zip_ref):
    """Extract education history."""
    rows = read_csv_from_zip(zip_ref, "Education.csv")
    if not rows:
        return ""
    
    lines = ["\n# EDUCATION", ""]
    
    for edu in rows:
        school = edu.get('School Name', 'Unknown School')
        degree = edu.get('Degree Name', '')
        field = edu.get('Field of Study', '')
        
        title = f"{degree} in {field}" if degree and field else degree or field or "Degree"
        
        start = edu.get('Start Date', '')
        end = edu.get('End Date', '')
        duration = f"{start} - {end}" if start else ""
        
        lines.append(f"## {school}")
        lines.append(f"{title}")
        if duration:
            lines.append(f"Duration: {duration}")
        
        if edu.get('Notes'):
            lines.append(f"\n{edu['Notes']}")
        
        if edu.get('Activities and Societies'):
            lines.append(f"Activities: {edu['Activities and Societies']}")
        
        lines.append("")
    
    return '\n'.join(lines)


def extract_skills(zip_ref):
    """Extract skills."""
    rows = read_csv_from_zip(zip_ref, "Skills.csv")
    if not rows:
        return ""
    
    skills = [row.get('Name', '') for row in rows if row.get('Name')]
    if not skills:
        return ""
    
    return f"\n# SKILLS\n\n{', '.join(skills)}\n"


def extract_certifications(zip_ref):
    """Extract certifications (newest first)."""
    rows = read_csv_from_zip(zip_ref, "Certifications.csv")
    if not rows:
        return ""
    
    # Sort by date (newest first)
    rows = sorted(rows, key=lambda x: x.get('Started On', ''), reverse=True)
    
    lines = ["\n# CERTIFICATIONS (Most Recent First)", ""]
    
    for cert in rows:
        name = cert.get('Name', 'Unknown Certification')
        authority = cert.get('Authority', '')
        date = cert.get('Started On', '')
        
        line = f"- {name}"
        if authority:
            line += f" ({authority})"
        if date:
            line += f" - {date}"
        lines.append(line)
    
    return '\n'.join(lines) + '\n'


def extract_projects(zip_ref):
    """Extract projects (newest first)."""
    rows = read_csv_from_zip(zip_ref, "Projects.csv")
    if not rows:
        return ""
    
    # Sort by start date (newest first)
    rows = sorted(rows, key=lambda x: x.get('Started On', ''), reverse=True)
    
    lines = ["\n# PROJECTS (Most Recent First)", ""]
    
    for proj in rows:
        name = proj.get('Title', 'Unknown Project')
        
        start = proj.get('Started On', '')
        end = proj.get('Finished On', '')
        duration = f"{start} - {end}" if start else ""
        
        lines.append(f"## {name}")
        if duration:
            lines.append(f"Duration: {duration}")
        
        if proj.get('Description'):
            lines.append(f"\n{proj['Description']}")
        
        if proj.get('Url'):
            lines.append(f"URL: {proj['Url']}")
        
        lines.append("")
    
    return '\n'.join(lines)


def extract_languages(zip_ref):
    """Extract languages."""
    rows = read_csv_from_zip(zip_ref, "Languages.csv")
    if not rows:
        return ""
    
    lines = ["\n# LANGUAGES", ""]
    
    for lang in rows:
        name = lang.get('Name', '')
        proficiency = lang.get('Proficiency', '')
        if name:
            line = f"- {name}"
            if proficiency:
                line += f" ({proficiency})"
            lines.append(line)
    
    return '\n'.join(lines) + '\n'


def extract_recommendations(zip_ref):
    """Extract recommendations received."""
    rows = read_csv_from_zip(zip_ref, "Recommendations_Received.csv")
    if not rows:
        return ""
    
    lines = ["\n# RECOMMENDATIONS", ""]
    
    for rec in rows:
        recommender = f"{rec.get('First Name', '')} {rec.get('Last Name', '')}".strip()
        company = rec.get('Company', '')
        title = rec.get('Job Title', '')
        text = rec.get('Text', '')
        
        if recommender:
            header = f"## From {recommender}"
            if title and company:
                header += f" ({title} at {company})"
            lines.append(header)
        
        if text:
            lines.append(f"\n\"{text}\"")
        
        lines.append("")
    
    return '\n'.join(lines)


def extract_linkedin_zip(zip_path):
    """Extract all relevant data from LinkedIn ZIP export."""
    
    print(f"📦 Processing LinkedIn export: {zip_path}")
    
    # Ensure output directory exists
    ME_DIR.mkdir(exist_ok=True)
    
    # Check if it's a nested zip (zip.zip)
    if str(zip_path).endswith('.zip.zip'):
        print("  📂 Detected nested ZIP, extracting outer layer...")
        with zipfile.ZipFile(zip_path, 'r') as outer_zip:
            # Find inner zip
            inner_zips = [n for n in outer_zip.namelist() if n.endswith('.zip')]
            if inner_zips:
                inner_zip_name = inner_zips[0]
                inner_zip_data = outer_zip.read(inner_zip_name)
                
                # Process inner zip from memory
                from io import BytesIO
                zip_buffer = BytesIO(inner_zip_data)
                return process_zip(zipfile.ZipFile(zip_buffer, 'r'))
    
    return process_zip(zipfile.ZipFile(zip_path, 'r'))


def process_zip(zip_ref):
    """Process the opened ZIP file."""
    with zip_ref:
        # List contents
        print("  📋 ZIP contents:")
        for name in sorted(zip_ref.namelist())[:20]:
            print(f"      - {name}")
        if len(zip_ref.namelist()) > 20:
            print(f"      ... and {len(zip_ref.namelist()) - 20} more files")
        
        # Extract all sections
        sections = []
        
        print("\n  🔍 Extracting profile data...")
        profile = extract_profile(zip_ref)
        if profile:
            sections.append(profile)
            print("      ✓ Profile")
        
        positions = extract_positions(zip_ref)
        if positions:
            sections.append(positions)
            print("      ✓ Work Experience")
        
        education = extract_education(zip_ref)
        if education:
            sections.append(education)
            print("      ✓ Education")
        
        skills = extract_skills(zip_ref)
        if skills:
            sections.append(skills)
            print("      ✓ Skills")
        
        certifications = extract_certifications(zip_ref)
        if certifications:
            sections.append(certifications)
            print("      ✓ Certifications")
        
        projects = extract_projects(zip_ref)
        if projects:
            sections.append(projects)
            print("      ✓ Projects")
        
        languages = extract_languages(zip_ref)
        if languages:
            sections.append(languages)
            print("      ✓ Languages")
        
        recommendations = extract_recommendations(zip_ref)
        if recommendations:
            sections.append(recommendations)
            print("      ✓ Recommendations")
        
        # Combine all sections
        full_text = '\n'.join(sections)
        
        # Write output
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"\n✅ Successfully extracted LinkedIn data to {OUTPUT_FILE}")
        print(f"   Total characters: {len(full_text):,}")
        
        return full_text


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        zip_path = find_linkedin_zip()
        if not zip_path:
            print("❌ No LinkedIn ZIP file found in me/ directory")
            print("   Please provide the path to your LinkedIn export:")
            print("   python scripts/extract_linkedin_zip.py <path_to_zip>")
            print("\n   Or place your LinkedIn export in the me/ folder")
            sys.exit(1)
    
    if not zip_path.exists():
        print(f"❌ File not found: {zip_path}")
        sys.exit(1)
    
    extract_linkedin_zip(zip_path)


if __name__ == "__main__":
    main()
