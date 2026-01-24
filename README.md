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
- **Auto-Updates**: Weekly GitHub Action automatically downloads and updates your LinkedIn profile
- **Contact Tracking**: NTFY notifications when someone shares their email
- **Easy to Use**: Gradio interface - simple, beautiful, and requires no frontend coding

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GROQ API key ([get one free](https://console.groq.com/))
- Your LinkedIn profile PDF
- (Optional) NTFY account for notifications

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
# Add your GOOGLE_API_KEY, NTFY credentials

# 4. Add your LinkedIn PDF
# Save your LinkedIn profile as me/linkedin.pdf

# 5. Run the app
python app.py
```

Your app will be running at http://localhost:7860

## 📁 Project Structure

```
my_career_agent/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── HOSTING.md            # Hosting options guide
├── DEPLOYMENT.md         # Deployment checklist
├── me/
│   ├── linkedin.pdf      # Your LinkedIn profile
│   └── summary.txt       # Your personal summary
├── utils/
│   └── llm_utils.py      # LLM helper functions
├── scripts/
│   └── download_linkedin.py  # LinkedIn automation (optional)
└── .github/
    └── workflows/
        └── update-linkedin.yml  # Weekly auto-update (optional)
```
