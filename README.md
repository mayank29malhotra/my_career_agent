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

## ☁️ HuggingFace Data Storage (Recommended for Production)

Store your LinkedIn data privately on HuggingFace instead of in the repo:

### Setup HuggingFace Dataset

1. **Create a private dataset** on HuggingFace:
   - Go to https://huggingface.co/new-dataset
   - Name it something like `career-agent-data`
   - Set visibility to **Private**

2. **Upload your linkedin.txt**:
   - After running `extract_linkedin_zip.py`, upload `me/linkedin.txt` to your dataset
   - You can do this weekly to keep it updated

3. **Configure Environment Variables**:
   ```bash
   HF_LINKEDIN_REPO=your-username/career-agent-data
   HF_LINKEDIN_FILE=linkedin.txt
   HF_TOKEN=hf_xxxxx  # Required for private repos
   ```

4. **For HuggingFace Spaces**: Add these as Space Secrets in your Space settings

### 🔄 Weekly Update Workflow

1. Download fresh LinkedIn data export from LinkedIn (repeat the steps above)
2. Run `python scripts/extract_linkedin_zip.py`
3. Upload the new `me/linkedin.txt` to your HuggingFace dataset (replacing the old file)
4. Your deployed agent will automatically use the latest data on next restart!

## 📁 Project Structure

```
my_career_agent/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── me/
│   ├── linkedin.txt      # Extracted LinkedIn data (gitignored)
│   └── summary.txt       # Your personal summary
├── utils/
│   └── llm_utils.py      # LLM helper functions
└── scripts/
    ├── extract_linkedin_zip.py  # Extract data from LinkedIn ZIP export
    └── load_linkedin_hf.py      # Load data from HuggingFace
```

## 🔒 Privacy & Security

Your LinkedIn data is **never pushed to GitHub**:
- `me/linkedin.txt` and `me/linkedin.pdf` are in `.gitignore`
- LinkedIn ZIP files are also gitignored
- Use HuggingFace **private** datasets for production deployment
- Your HuggingFace token should be stored as an environment secret

## 🔧 Data Loading Priority

The app loads LinkedIn data in this order:
1. **Local file**: `me/linkedin.txt` (fastest, for development)
2. **HuggingFace**: Downloads from your private dataset (for production)
3. **PDF fallback**: `me/linkedin.pdf` (legacy support)

## 🛠️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your GROQ API key for LLM |
| `NTFY_TOPIC` | No | NTFY topic for notifications |
| `HF_LINKEDIN_REPO` | No* | HuggingFace dataset ID (e.g., `username/career-agent-data`) |
| `HF_LINKEDIN_FILE` | No | Filename in HF dataset (default: `linkedin.txt`) |
| `HF_TOKEN` | No* | HuggingFace token for private repos |

*Required for HuggingFace-based deployment

## 🤔 FAQ

**Q: Why use LinkedIn ZIP export instead of PDF?**
A: The ZIP export contains structured CSV data with all your information - work history, skills, certifications, recommendations, projects, and more. It's much more comprehensive than the PDF.

**Q: How often should I update my LinkedIn data?**
A: Weekly is recommended to keep your agent up-to-date with any profile changes.

**Q: Is my LinkedIn data safe?**
A: Yes! The data is never committed to Git (gitignored), and when using HuggingFace, you should use a private dataset.

**Q: Can I use this without HuggingFace?**
A: Absolutely! For local development, just place your `linkedin.txt` in the `me/` folder. For deployment, you'll need some way to provide the data file.
