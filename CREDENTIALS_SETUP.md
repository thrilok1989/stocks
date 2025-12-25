# 🔑 How to Setup Your API Credentials

## 📋 Quick Setup (3 Steps)

### **Step 1: Copy the template file**

```bash
cd /home/user/stocks
cp .env.example .env
```

This creates your personal `.env` file (which is gitignored - safe!)

---

### **Step 2: Edit .env and add your credentials**

```bash
nano .env
```

**Or open with any text editor**

Replace the placeholder values with your actual credentials:

```bash
# Before (template):
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here

# After (your actual values):
DHAN_CLIENT_ID=1100123456
DHAN_ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIx...
```

**Save the file!** (Ctrl+X, then Y, then Enter in nano)

---

### **Step 3: Install python-dotenv** (if not already installed)

```bash
pip install python-dotenv
```

---

### **Step 4: Run the app**

```bash
python3 flask_backend.py
```

**Done!** Your credentials are loaded automatically! ✅

---

## 🔐 What Credentials Do You Need?

### **Required (Minimum to run):**

| Credential | Required? | Where to Get |
|------------|-----------|--------------|
| **DHAN_CLIENT_ID** | ✅ Required | DhanHQ Dashboard |
| **DHAN_ACCESS_TOKEN** | ✅ Required | DhanHQ API Section |

### **Optional (For Extra Features):**

| Credential | Optional? | What It Enables |
|------------|-----------|-----------------|
| **TELEGRAM_BOT_TOKEN** | ⭐ Optional | Trading alerts on Telegram |
| **TELEGRAM_CHAT_ID** | ⭐ Optional | Trading alerts on Telegram |
| **NEWSDATA_API_KEY** | ⭐ Optional | News sentiment analysis |
| **PERPLEXITY_API_KEY** | ⭐ Optional | AI market insights |

---

## 📝 Where to Get Each Credential

### **1. DhanHQ Credentials** (REQUIRED)

```
1. Login to: https://api.dhan.co
2. Go to "API" section
3. Copy:
   - Client ID (like: 1100123456)
   - Access Token (generate new if needed)
```

**Paste these in .env file**

---

### **2. Telegram Bot** (OPTIONAL - For Alerts)

```
1. Open Telegram
2. Search for: @BotFather
3. Send: /newbot
4. Follow instructions
5. Copy the token (like: 123456:ABC-DEF...)

6. Get Chat ID:
   - Send message to your bot
   - Go to: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   - Copy the "chat":{"id": number}
```

**Paste both in .env file**

---

### **3. NewsData API** (OPTIONAL - For News)

```
1. Go to: https://newsdata.io
2. Sign up (free tier available)
3. Copy API key from dashboard
```

---

### **4. Perplexity API** (OPTIONAL - For AI)

```
1. Go to: https://www.perplexity.ai/settings/api
2. Generate API key
3. Copy it
```

---

## ✅ Verify Your Setup

After adding credentials, verify they work:

```bash
# Run the app
python3 flask_backend.py

# Open browser
http://localhost:5000

# Check:
# ✅ Backend status shows "Connected" (green)
# ✅ Market data loads (NIFTY, SENSEX prices)
# ✅ No error messages in terminal
```

---

## 🔒 Security Notes

### **✅ SAFE:**
- `.env` file is gitignored (won't be committed)
- Credentials stay on your local machine
- Never shared publicly

### **❌ NEVER:**
- Don't commit `.env` to GitHub
- Don't share `.env` file with others
- Don't hardcode credentials in Python files
- Don't post credentials in screenshots

---

## 🐛 Troubleshooting

### **Issue: "No module named 'dotenv'"**

**Solution:**
```bash
pip install python-dotenv
```

### **Issue: Credentials not loading**

**Solution:**
```bash
# Make sure .env file exists
ls -la .env

# Make sure it has actual values (not placeholders)
cat .env

# Restart the app
python3 flask_backend.py
```

### **Issue: "Invalid credentials" error**

**Solution:**
- Check DhanHQ access token is current (they expire!)
- Generate new token from DhanHQ dashboard
- Update .env file with new token
- Restart app

---

## 📂 File Structure

```
stocks/
├── .env.example          ← Template (committed to GitHub)
├── .env                  ← Your actual secrets (gitignored - safe!)
├── flask_backend.py      ← Loads from .env automatically
├── .gitignore            ← Contains .env (secrets protected)
└── CREDENTIALS_SETUP.md  ← This file
```

---

## 🎯 Quick Reference

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env

# Install dotenv
pip install python-dotenv

# Run app
python3 flask_backend.py

# Open browser
http://localhost:5000
```

---

## ✅ Summary

1. ✅ Copy `.env.example` to `.env`
2. ✅ Add your DhanHQ credentials to `.env`
3. ✅ Optionally add Telegram/News/AI keys
4. ✅ Run `python3 flask_backend.py`
5. ✅ Start trading!

**Your secrets are safe and never committed to GitHub!** 🔒

---

**Need help getting any credential? Just ask!** 🚀
