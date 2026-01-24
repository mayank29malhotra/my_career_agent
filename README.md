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

### Deploy Always-On (Choose One)

See **[HOSTING.md](HOSTING.md)** for detailed hosting options:

#### 🏆 Easiest: Hugging Face + Keep-Alive (FREE)
1. Already deployed to HF? ✅
2. Go to [UptimeRobot.com](https://uptimerobot.com)
3. Add monitor with your HF Space URL
4. Set interval to 5 minutes
5. Done! Your app never sleeps 🎉

#### 🥇 Best Free: Oracle Cloud (FREE Forever)
- True always-on, FREE forever
- Full VM control
- Setup time: 30 minutes
- [See HOSTING.md for steps](HOSTING.md#-alternative-4-oracle-cloud-free-tier-best-free)

#### 🥈 Best UX: Railway ($5/month credit)
```powershell
npm install -g @railway/cli
railway login
railway init
railway up
```
- $5 monthly credit (enough for 24/7)
- Great developer experience
- [See HOSTING.md for details](HOSTING.md#-alternative-1-railwayapp-free-5month-credit)

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

## 🔐 LinkedIn Automation Setup

To enable the automated LinkedIn PDF download (locally or via GitHub Actions), set these environment variables:

- LINKEDIN_EMAIL: Your LinkedIn login email
- LINKEDIN_PASSWORD: Your LinkedIn password
- LINKEDIN_PROFILE_URL: Your profile URL, e.g. https://www.linkedin.com/in/your-handle/
- LINKEDIN_SKIP_LOGIN: Set to `true` to attempt public profile download without logging in (recommended to reduce account risk). If public profile is not visible, the script will exit with a helpful message.

GitHub Actions:
- Add `LINKEDIN_EMAIL`, `LINKEDIN_PASSWORD`, and `LINKEDIN_PROFILE_URL` as repository secrets
- The workflow will consume them when running `scripts/download_linkedin.py`

## 🔧 How It Works

### Gradio Application
- **Backend**: Python with Gradio framework
- **AI**: GROQ API for conversations
- **UI**: Gradio's built-in chat interface (no frontend coding needed)
- **Hosting**: Multiple options (see [HOSTING.md](HOSTING.md))

### Always-On Strategy
| Service | Cost | Uptime | Setup |
|---------|------|--------|-------|
| HF + UptimeRobot | FREE ✅ | 99%+ | 5 min |
| Oracle Cloud | FREE ✅ | 100% | 30 min |
| Railway | $5 credit/mo | 100% | 10 min |
| Render | FREE ✅ | 99%+ | 5 min |
| Fly.io | ~$2/mo | 100% | 15 min |

See [HOSTING.md](HOSTING.md) for detailed comparison.

**Recommended for most users:** HF + UptimeRobot (completely FREE)e VM hosting forever
4. **Your own VPS**: Full control

See [HOSTING.md](HOSTING.md) for detailed comparison.

## 💰 Cost Analysis

- **GitHub Pages**: FREE ✅
- **GitHub Actions**: 2,000 min/month free ✅
- **GROQ API**: Free tier available ✅
- **Total**: $0/month 🎉

## 🐛 Troubleshooting

### "API key not found" Error
- Check `.env` file exists with `GOOGLE_API_KEY`
- Verify the key is valid at https://aistudio.google.com/

### App Sleeping/Not Responding
- Set up keep-alive service (see [HOSTING.md](HOSTING.md))
- Use UptimeRobot to ping every 5 minutes
- Or switch to always-on hosting (Railway, Oracle Cloud)

### LinkedIn Download Fails
- Check GitHub Actions logs
- Verify credentials in GitHub Secrets
- Try running locally: `python scripts/download_linkedin.py`
- Check for captcha or 2FA requirements
- Confirm `LINKEDIN_PROFILE_URL` is set and correct (public profile URL)
- If using `LINKEDIN_SKIP_LOGIN=true`, ensure your profile’s “Public profile visibility” is enabled in LinkedIn settings. Otherwise LinkedIn will redirect to an auth wall and the script will stop.

### Gradio Not Starting
- Verify port 7860 is available
- Check all dependencies installed: `pip install -r requirements.txt`
- Look for errors in console output

## 📝 Original vs Production

| Feature | Hugging Face | GitHub Pages |
|---------|--------------|--------------|
| Hosting | Gradio | Static HTML/JS |
| Cost | Free (sleeps) | Free (24/7) |
| Uptime | Sleep mode | Always on |
| Updates | Manual | Auto (weekly) |

---

**Built with ❤️ using GROQ, GitHub Pages, and GitHub Actions**

