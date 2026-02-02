---
title: my_career_agent
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.9"
app_file: app.py
pinned: false
---

# 🤖 My Career Agent - Production Ready

An AI-powered career agent that acts as your alter ego, answering questions about your professional background based on your LinkedIn profile. Perfect for recruiters, potential employers, and collaborators to learn about you 24/7.

## ✨ Features

- **True Alter Ego**: The AI doesn't just represent you—it embodies your persona, speaking in first person with your personality
- **Smart Conversations**: Powered by GROQ: llama-3.3-70b-versatile, trained on your LinkedIn profile and personal summary
- **Always Online**: Multiple free/cheap hosting options that stay active 24/7
- **LinkedIn ZIP Export Support**: Extracts data from LinkedIn's official data export (no automation needed!)
- **HuggingFace Integration**: Store your LinkedIn data privately on HuggingFace datasets
- **Manual Weekly Updates**: Download your LinkedIn data weekly and upload to HuggingFace
- **Contact Tracking**: NTFY notifications when someone shares their email
- **Easy to Use**: Gradio interface - simple, beautiful, and requires no frontend coding

> ⚠️ **Note**: Automated LinkedIn profile updates via GitHub Actions have been removed. LinkedIn's security measures make automated downloads unreliable. Instead, manually download your data export weekly and upload to HuggingFace.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GROQ API key ([get one free](https://console.groq.com/))
- Your LinkedIn data export (see instructions below)
- (Optional) HuggingFace account for private data storage

### 📥 Download Your LinkedIn Data

LinkedIn allows you to download a complete export of your profile data:

1. **Go to LinkedIn Settings**
   - Click your profile picture → Settings & Privacy
   - Or visit: https://www.linkedin.com/mypreferences/d/download-my-data

2. **Request Your Data**
   - Select "Want something in particular?" 
   - Check the boxes you want (recommended: all profile data)
   - Click "Request archive"

3. **Wait for Email**
   - LinkedIn will email you when your data is ready (usually within 24 hours)
   - Download the ZIP file from the link in the email

4. **Extract Your Data**
   - Place the ZIP file in the `me/` folder
   - Run: `python scripts/extract_linkedin_zip.py`
   - This creates `me/linkedin.txt` with your formatted profile

> 💡 **Tip**: LinkedIn data exports include your full profile, work history, skills, certifications, recommendations, and more - much richer than a PDF!

### Local Setup

```powershell
# 1. Clone repository
git clone https://github.com/yourusername/my_career_agent.git
cd my_career_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
Copy-Item .env.example .env
notepad .env
# Add your GROQ_API_KEY, NTFY credentials

# 4. Download and extract your LinkedIn data
# Place your LinkedIn ZIP in me/ folder, then:
python scripts/extract_linkedin_zip.py

# 5. Run the app
python app.py
```

Your app will be running at http://localhost:7860

## ☁️ Deploying to HuggingFace Spaces

Upload your LinkedIn data directly to the Space repo's `me/` folder (alongside `summary.txt`):

### 🔄 Weekly Update Workflow

1. **Download** fresh LinkedIn data export from LinkedIn
2. **Extract** locally: `python scripts/extract_linkedin_zip.py`
3. **Upload** `me/linkedin.txt` to your HuggingFace Space:
   - Go to your Space → Files → `me/` folder
   - Click "Add file" → "Upload files"
   - Upload `linkedin.txt` (replacing the old one)
   - Commit changes
4. Your Space will automatically restart with the updated data!

```
Your HuggingFace Space:
├── app.py
├── requirements.txt
├── me/
│   ├── linkedin.txt    ← Upload here weekly
│   └── summary.txt
└── ...
```

> ⚠️ **Note**: `linkedin.txt` is gitignored locally but you upload it directly to HuggingFace. This keeps your personal data off GitHub while still being available in production.

## 📁 Project Structure

```
my_career_agent/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── me/
│   ├── linkedin.txt      # Extracted LinkedIn data (gitignored locally, uploaded to HF)
│   └── summary.txt       # Your personal summary
├── utils/
│   └── llm_utils.py      # LLM helper functions
└── scripts/
    └── extract_linkedin_zip.py  # Extract data from LinkedIn ZIP export
```

## 🔒 Privacy & Security

Your LinkedIn data is **never pushed to GitHub**:
- `me/linkedin.txt` and `me/linkedin.pdf` are in `.gitignore`
- LinkedIn ZIP files are also gitignored
- For production, upload `linkedin.txt` directly to your HuggingFace Space repo

## 🛠️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your GROQ API key for LLM |
| `NTFY_TOPIC` | No | NTFY topic for push notifications |

## 🤔 FAQ

**Q: Why use LinkedIn ZIP export instead of PDF?**
A: The ZIP export contains structured CSV data with all your information - work history, skills, certifications, recommendations, projects, and more. It's much more comprehensive than the PDF.

**Q: How often should I update my LinkedIn data?**
A: Weekly is recommended to keep your agent up-to-date with any profile changes.

**Q: Is my LinkedIn data safe?**
A: Yes! The data is never committed to GitHub (gitignored). You upload it directly to your HuggingFace Space.

**Q: Where do I upload linkedin.txt for production?**
A: Upload it directly to the `me/` folder in your HuggingFace Space repo (same place as `summary.txt`).

