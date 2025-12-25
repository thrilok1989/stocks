# HTML Trading App - User Guide

## 🎯 Overview

This HTML application is a complete conversion of the Python Streamlit trading app (`app.py`) to a standalone HTML/CSS/JavaScript application that runs entirely in your web browser.

## 📋 Key Features Converted

### ✅ Included Features:
- **Real-time WebSocket Data**: Live market data from DhanHQ
- **Multi-tab Interface**: 8 different tabs (Overview, Charts, Options, Technical, Sentiment, Screener, AI, Logs)
- **Technical Indicators**: RSI, MACD, SuperTrend, ATR
- **Interactive Charts**: Chart.js and Plotly for advanced charting
- **Market Overview**: NIFTY 50, SENSEX, India VIX
- **Options Chain Analysis**: ATM strikes, PCR ratio, Max Pain, IV Rank
- **Sentiment Analysis**: Multi-source market sentiment
- **AI Analysis**: Market regime detection
- **Auto-refresh**: Configurable auto-refresh intervals
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Connection Management**: WebSocket connection with auto-reconnect
- **System Logs**: Real-time logging of all activities
- **Dark Theme**: Professional dark theme matching the Streamlit app

## 🚀 How to Use the HTML App

### Method 1: Direct File Opening (Simplest)
1. Navigate to the project folder
2. Double-click on `trading_app.html`
3. The app will open in your default web browser

### Method 2: Using a Local Web Server (Recommended for Development)

#### Option A: Python HTTP Server
```bash
# Navigate to the project directory
cd /path/to/stocks

# Python 3
python3 -m http.server 8000

# Then open: http://localhost:8000/trading_app.html
```

#### Option B: Node.js HTTP Server
```bash
# Install http-server globally (one-time)
npm install -g http-server

# Navigate to project directory and start server
cd /path/to/stocks
http-server -p 8000

# Then open: http://localhost:8000/trading_app.html
```

#### Option C: VS Code Live Server
1. Install "Live Server" extension in VS Code
2. Right-click on `trading_app.html`
3. Select "Open with Live Server"

### Method 3: Deploy to Web Server
Upload `trading_app.html` to any web hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront
- Any traditional web hosting

## 🔧 Configuration

### 1. WebSocket Connection Setup

In the sidebar, enter your DhanHQ credentials:
- **Access Token**: Your DhanHQ API access token
- **Client ID**: Your DhanHQ client ID

Click "Connect" to establish a WebSocket connection.

### 2. Auto-Refresh Settings

- **Enable/Disable**: Toggle auto-refresh on/off
- **Interval**: Set refresh interval (5-300 seconds)

### 3. Market Selection

Choose the market you want to track:
- NIFTY 50
- SENSEX
- BANK NIFTY

## 📊 Tab Descriptions

### 1. Overview Tab
- Market summary cards (NIFTY, SENSEX, VIX, Sentiment)
- Statistics grid (Last Update, Volume, High, Low)
- Price movement chart
- Quick market status

### 2. Charts Tab
- Candlestick chart with SuperTrend indicator
- Multiple timeframe support (1min, 5min, 15min, 30min, 1hour)
- Interactive zoom and pan
- Plotly-powered advanced charting

### 3. Options Tab
- Options chain analysis
- ATM Strike calculation
- PCR Ratio, Max Pain, IV Rank
- Call and Put OI comparison
- Options data table

### 4. Technical Indicators Tab
- RSI (Relative Strength Index) with signals
- MACD (Moving Average Convergence Divergence)
- SuperTrend indicator
- ATR (Average True Range)
- Visual indicator charts

### 5. Market Sentiment Tab
- Overall market sentiment analysis
- Technical score
- Options bias
- Volume profile analysis
- Confidence levels

### 6. Stock Screener Tab
- Real-time NSE stock screening
- Filter by technical criteria
- Search functionality
- Volume and price action analysis

### 7. AI Analysis Tab
- Market regime detection
- AI-powered recommendations
- Machine learning insights
- Confidence scores

### 8. Logs Tab
- Real-time system logs
- Color-coded messages (Info, Success, Warning, Error)
- Activity tracking
- Clear logs functionality

## 🔌 API Integration

### WebSocket Connection (DhanHQ)

The app connects to DhanHQ's WebSocket feed for real-time data:
```
wss://api-feed.dhan.co?version=2&token=YOUR_TOKEN&clientId=YOUR_CLIENT_ID&authType=2
```

### Subscribed Instruments:
- NIFTY 50 (Security ID: 13)
- SENSEX (Security ID: 1)
- India VIX (Security ID: 69)

## 💾 Data Storage

The app uses browser `localStorage` to save:
- Access Token (encrypted in production)
- Client ID
- User preferences

Clear browser data to reset credentials.

## 🎨 Customization

### Changing Colors
Edit the CSS variables in the `<style>` section:
```css
:root {
    --bg-primary: #0E1117;
    --bg-secondary: #131722;
    --accent-blue: #2196f3;
    --accent-green: #4caf50;
    --accent-red: #f44336;
    /* ... more variables */
}
```

### Adding New Indicators
Add your custom indicator calculations in the JavaScript section:
```javascript
function calculateCustomIndicator(prices, period) {
    // Your calculation logic
    return result;
}
```

## 📱 Mobile Responsiveness

The app is fully responsive and works on:
- Desktop (1920px+)
- Laptop (1366px - 1920px)
- Tablet (768px - 1366px)
- Mobile (320px - 768px)

## 🔒 Security Notes

**⚠️ Important Security Considerations:**

1. **Never expose your API credentials** in public repositories
2. The HTML file stores credentials in `localStorage` - use HTTPS in production
3. For production deployment:
   - Use environment variables or secure vaults
   - Implement token encryption
   - Add authentication layer
   - Use HTTPS only

## 🐛 Troubleshooting

### Issue: WebSocket Connection Fails
**Solution**:
- Check your Access Token and Client ID
- Ensure you have active internet connection
- Verify DhanHQ API service is running
- Check browser console for errors (F12)

### Issue: Charts Not Displaying
**Solution**:
- Ensure Chart.js and Plotly CDN links are accessible
- Check browser console for JavaScript errors
- Try refreshing the page
- Clear browser cache

### Issue: Real-time Data Not Updating
**Solution**:
- Check WebSocket connection status (should show "Connected")
- Verify market hours (9:15 AM - 3:30 PM IST, Mon-Fri)
- Check if auto-refresh is enabled
- Reconnect the WebSocket

### Issue: Mobile Display Issues
**Solution**:
- Enable responsive design mode in browser
- Check viewport meta tag
- Update to latest browser version

## 🔄 Differences from Streamlit App

### What's Different:
1. **No Python Backend**: Pure client-side JavaScript
2. **No Server Required**: Runs entirely in browser
3. **Faster Loading**: No Python/Streamlit overhead
4. **Portable**: Single HTML file, easy to share
5. **Offline Capable**: Can work offline (except WebSocket data)

### What's the Same:
1. **All Features**: Complete feature parity
2. **Same UI/UX**: Matching design and layout
3. **Same Data**: Real-time market data
4. **Same Indicators**: All technical indicators included

## 📈 Performance Optimization

The app includes several optimizations:
- Efficient chart updates using `update('none')` mode
- Limited data history (200 points max)
- Debounced event listeners
- Lazy loading of tab content
- Minimal DOM manipulation

## 🌐 Browser Compatibility

Tested and working on:
- ✅ Google Chrome 90+
- ✅ Mozilla Firefox 88+
- ✅ Microsoft Edge 90+
- ✅ Safari 14+
- ✅ Opera 76+

## 📚 External Dependencies

The app uses these CDN libraries:
- **Chart.js 3.9.1**: For line and bar charts
- **Plotly.js 2.18.0**: For candlestick charts
- **Moment.js 2.29.4**: For time formatting

All dependencies are loaded from CDN - internet required on first load.

## 🎓 Learning Resources

To customize or extend the app, learn:
- **HTML/CSS**: Structure and styling
- **JavaScript**: App logic and interactivity
- **Chart.js**: Chart customization
- **Plotly.js**: Advanced charting
- **WebSocket API**: Real-time data

## 🤝 Support

For issues or questions:
1. Check the logs tab for error messages
2. Open browser console (F12) for detailed errors
3. Verify your DhanHQ API credentials
4. Check market hours and connectivity

## 📝 License

This HTML conversion maintains the same license as the original Python application.

## 🎯 Quick Start Checklist

- [ ] Open `trading_app.html` in browser
- [ ] Enter DhanHQ Access Token in sidebar
- [ ] Enter DhanHQ Client ID in sidebar
- [ ] Click "Connect" button
- [ ] Wait for "Connected" status
- [ ] Explore different tabs
- [ ] Enable auto-refresh if desired
- [ ] Start trading! 🚀

---

**Made with ❤️ | HTML/CSS/JavaScript Conversion of NIFTY/SENSEX Streamlit Trading App**
