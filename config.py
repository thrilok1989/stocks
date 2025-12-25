"""
Configuration settings for NIFTY/SENSEX Manual Trader
All sensitive credentials are loaded from environment variables (.env file)
"""

import pytz
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════════
# TIMEZONE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Indian Standard Time (IST) - Use this for all datetime operations
IST = pytz.timezone('Asia/Kolkata')

def get_current_time_ist():
    """Get current time in IST timezone"""
    return datetime.now(IST)

# ═══════════════════════════════════════════════════════════════════════
# CREDENTIALS - Loaded from Environment Variables (.env file)
# ═══════════════════════════════════════════════════════════════════════

def get_dhan_credentials():
    """Load DhanHQ credentials from environment variables"""
    try:
        client_id = os.getenv("DHAN_CLIENT_ID", "")
        access_token = os.getenv("DHAN_ACCESS_TOKEN", "")

        if not client_id or client_id == "your_client_id_here":
            print(f"⚠️ DhanHQ CLIENT_ID not configured in .env file")
            return None

        if not access_token or access_token == "your_access_token_here":
            print(f"⚠️ DhanHQ ACCESS_TOKEN not configured in .env file")
            return None

        return {
            'client_id': client_id,
            'access_token': access_token,
            'api_key': os.getenv("DHAN_API_KEY", ""),
            'api_secret': os.getenv("DHAN_API_SECRET", "")
        }
    except Exception as e:
        print(f"⚠️ DhanHQ credentials error: {e}")
        return None

def get_telegram_credentials():
    """Load Telegram credentials from environment variables"""
    try:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not bot_token or bot_token == "your_telegram_bot_token_here":
            return {'enabled': False}

        if not chat_id or chat_id == "your_telegram_chat_id_here":
            return {'enabled': False}

        return {
            'bot_token': bot_token,
            'chat_id': chat_id,
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ Telegram credentials error: {e}")
        return {'enabled': False}

# ═══════════════════════════════════════════════════════════════════════
# AI CONFIGURATION - Loaded from Environment Variables (.env file)
# ═══════════════════════════════════════════════════════════════════════

def get_perplexity_credentials():
    """Load Perplexity credentials from environment variables"""
    try:
        api_key = os.getenv("PERPLEXITY_API_KEY", "")

        if not api_key or api_key == "your_perplexity_api_key_here":
            return {'enabled': False}

        return {
            'api_key': api_key,
            'model': os.getenv("PERPLEXITY_MODEL", "sonar"),
            'search_depth': os.getenv("PERPLEXITY_SEARCH_DEPTH", "medium"),
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ Perplexity credentials error: {e}")
        return {'enabled': False}

def get_newsdata_credentials():
    """Load NewsData credentials from environment variables"""
    try:
        api_key = os.getenv("NEWSDATA_API_KEY", "")

        if not api_key or api_key == "your_newsdata_api_key_here":
            return {'enabled': False}

        return {
            'api_key': api_key,
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ NewsData credentials error: {e}")
        return {'enabled': False}

# AI Settings
AI_RUN_ONLY_DIRECTIONAL = os.environ.get("AI_RUN_ONLY_DIRECTIONAL", "") == "1"
AI_REPORT_DIR = os.environ.get("AI_REPORT_DIR", "ai_reports")

# ═══════════════════════════════════════════════════════════════════════
# MARKET HOURS SETTINGS (All times in IST - Indian Standard Time)
# ═══════════════════════════════════════════════════════════════════════

MARKET_HOURS_ENABLED = True  # Set to False to disable market hours checking

# Market session timings (IST)
MARKET_HOURS = {
    'pre_market_open': '08:30',    # 8:30 AM IST
    'market_open': '09:15',        # 9:15 AM IST
    'market_close': '15:30',       # 3:30 PM IST
    'post_market_close': '15:45'   # 3:45 PM IST (App will run until this time)
}

# Session-based refresh intervals (seconds)
# Optimized to prevent API rate limiting (HTTP 429)
REFRESH_INTERVALS = {
    'pre_market': 60,      # 60 seconds during pre-market
    'regular': 60,         # 60 seconds during regular trading
    'post_market': 60,     # 60 seconds during post-market
    'closed': 60           # 60 seconds when market is closed (1 minute)
}

# ═══════════════════════════════════════════════════════════════════════
# TRADING SETTINGS
# ═══════════════════════════════════════════════════════════════════════

LOT_SIZES = {
    "NIFTY": 75,
    "SENSEX": 30
}

STRIKE_INTERVALS = {
    "NIFTY": 50,
    "SENSEX": 100
}

SENSEX_NIFTY_RATIO = 3.3  # SENSEX ≈ 3.3 × NIFTY

STOP_LOSS_OFFSET = 10  # Points
SIGNALS_REQUIRED = 3
VOB_TOUCH_TOLERANCE = 5  # Points

# ═══════════════════════════════════════════════════════════════════════
# UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# Auto-refresh interval: 1 minute (60 seconds)
AUTO_REFRESH_INTERVAL = 60  # seconds (1 minute - optimized for fast clicks)

# Spot price refresh interval: 10 seconds for real-time updates
SPOT_PRICE_REFRESH_INTERVAL = 10  # seconds (10 seconds for spot price only)

DEMO_MODE = False

APP_TITLE = "🎯 NIFTY/SENSEX Manual Trader"
APP_SUBTITLE = "VOB-Based Trading | Manual Signal Entry"

COLORS = {
    'bullish': '#089981',
    'bearish': '#f23645',
    'neutral': '#787B86'
}

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_all_credentials():
    """Get all credentials in a single call"""
    return {
        'dhan': get_dhan_credentials(),
        'telegram': get_telegram_credentials(),
        'perplexity': get_perplexity_credentials(),
        'newsdata': get_newsdata_credentials()
    }

def is_ai_enabled():
    """Check if AI features are enabled"""
    perplexity_creds = get_perplexity_credentials()
    newsdata_creds = get_newsdata_credentials()
    return perplexity_creds.get('enabled', False) and newsdata_creds.get('enabled', False)
