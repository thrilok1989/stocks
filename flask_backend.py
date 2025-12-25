"""
Flask Backend Server for NIFTY/SENSEX Trading Dashboard
Uses all existing logic from app.py as backend
Serves HTML frontend via API endpoints
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
from datetime import datetime
import asyncio
import threading

# Import all modules from existing app
from config import *
from market_data import *
from market_hours_scheduler import scheduler, is_within_trading_hours, should_run_app, get_market_status
from signal_manager import SignalManager
from strike_calculator import calculate_strike, calculate_levels
from trade_executor import TradeExecutor
from telegram_alerts import TelegramBot
from bias_analysis import BiasAnalysisPro
from advanced_chart_analysis import AdvancedChartAnalysis
from overall_market_sentiment import calculate_overall_sentiment, run_ai_analysis
from data_cache_manager import (
    get_cache_manager,
    preload_all_data,
    get_cached_nifty_data,
    get_cached_sensex_data,
    get_cached_bias_analysis_results
)
from vob_signal_generator import VOBSignalGenerator
from htf_sr_signal_generator import HTFSRSignalGenerator

# Initialize Flask app
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize managers (similar to Streamlit session state)
class AppState:
    def __init__(self):
        self.signal_manager = SignalManager()
        self.vob_signal_generator = VOBSignalGenerator(proximity_threshold=8.0)
        self.htf_sr_signal_generator = HTFSRSignalGenerator(proximity_threshold=8.0)
        self.bias_analyzer = None  # Lazy load
        self.chart_analyzer = None  # Lazy load
        self.active_vob_signals = []
        self.active_htf_sr_signals = []
        self.active_positions = {}
        self.cached_sentiment = None
        self.last_ai_analysis_time = 0

state = AppState()

# Preload data on startup
preload_all_data()

# ============================================================================
# API ROUTES - Market Data
# ============================================================================

@app.route('/')
def serve_frontend():
    """Serve the main HTML frontend"""
    return send_from_directory('.', 'trading_dashboard_frontend.html')

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """Get current market data for NIFTY, SENSEX, VIX"""
    try:
        nifty_data = get_cached_nifty_data()
        sensex_data = get_cached_sensex_data()

        # Get cache manager for freshness
        cache_manager = get_cache_manager()
        nifty_cache_time = cache_manager._cache_timestamps.get('nifty_data', 0)

        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'nifty': nifty_data,
                'sensex': sensex_data,
                'cache_age': int(time.time() - nifty_cache_time) if nifty_cache_time > 0 else None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/market-status', methods=['GET'])
def market_status():
    """Get current market status (open/closed/pre-market/post-market)"""
    try:
        status = get_market_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - Sentiment Analysis
# ============================================================================

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    """Get overall market sentiment analysis"""
    try:
        nifty_data = get_cached_nifty_data()
        sensex_data = get_cached_sensex_data()

        # Calculate overall sentiment
        sentiment_result = calculate_overall_sentiment(
            nifty_data,
            sensex_data,
            state.active_vob_signals,
            state.active_htf_sr_signals
        )

        state.cached_sentiment = sentiment_result

        return jsonify({
            'success': True,
            'sentiment': sentiment_result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - Signal Management
# ============================================================================

@app.route('/api/signals/create', methods=['POST'])
def create_signal_setup():
    """Create a new signal setup"""
    try:
        data = request.json
        index = data.get('index')
        direction = data.get('direction')
        vob_support = float(data.get('vob_support'))
        vob_resistance = float(data.get('vob_resistance'))

        signal_id = state.signal_manager.create_setup(
            index, direction, vob_support, vob_resistance
        )

        return jsonify({
            'success': True,
            'signal_id': signal_id,
            'message': 'Signal setup created successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/active', methods=['GET'])
def get_active_signals():
    """Get all active signal setups"""
    try:
        active_setups = state.signal_manager.get_active_setups()
        return jsonify({
            'success': True,
            'signals': active_setups
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/<signal_id>/add', methods=['POST'])
def add_signal(signal_id):
    """Add a confirmation signal to setup"""
    try:
        state.signal_manager.add_signal(signal_id)

        # Check if ready and send Telegram
        setup = state.signal_manager.get_setup(signal_id)
        if setup and setup['status'] == 'ready':
            telegram = TelegramBot()
            telegram.send_signal_ready(setup)

        return jsonify({
            'success': True,
            'message': 'Signal added',
            'setup': setup
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/<signal_id>/remove', methods=['POST'])
def remove_signal(signal_id):
    """Remove a confirmation signal"""
    try:
        state.signal_manager.remove_signal(signal_id)
        setup = state.signal_manager.get_setup(signal_id)
        return jsonify({
            'success': True,
            'message': 'Signal removed',
            'setup': setup
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals/<signal_id>/delete', methods=['DELETE'])
def delete_signal(signal_id):
    """Delete a signal setup"""
    try:
        state.signal_manager.delete_setup(signal_id)
        return jsonify({
            'success': True,
            'message': 'Signal setup deleted'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - Trade Execution
# ============================================================================

@app.route('/api/trade/execute', methods=['POST'])
def execute_trade():
    """Execute a trade from signal setup"""
    try:
        data = request.json
        signal_id = data.get('signal_id')

        setup = state.signal_manager.get_setup(signal_id)
        if not setup:
            return jsonify({'success': False, 'error': 'Signal setup not found'}), 404

        if setup['signal_count'] < 3:
            return jsonify({'success': False, 'error': 'Need 3 confirmations'}), 400

        # Get current market data
        if setup['index'] == 'NIFTY':
            market_data = get_cached_nifty_data()
        else:
            market_data = get_cached_sensex_data()

        # Execute trade
        executor = TradeExecutor()
        result = executor.execute_trade(
            setup,
            market_data['spot_price'],
            market_data['current_expiry']
        )

        if result['success']:
            # Mark as executed
            state.signal_manager.mark_executed(signal_id, result['order_id'])

            # Store position
            details = result['order_details']
            state.active_positions[result['order_id']] = {
                'order_id': result['order_id'],
                'index': setup['index'],
                'direction': setup['direction'],
                'strike': details['strike'],
                'option_type': details['option_type'],
                'quantity': details['quantity'],
                'entry': details['entry_level'],
                'sl': details['sl_price'],
                'target': details['target_price'],
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }

            # Send Telegram notification
            telegram = TelegramBot()
            telegram.send_trade_executed(state.active_positions[result['order_id']])

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get all active positions"""
    try:
        positions = list(state.active_positions.values())

        # Calculate current P&L for each position
        for pos in positions:
            if pos['status'] == 'active':
                market_data = get_cached_nifty_data() if pos['index'] == 'NIFTY' else get_cached_sensex_data()
                current_price = market_data['spot_price']

                if pos['direction'] == 'CALL':
                    pos['current_pnl'] = (current_price - pos['entry']) * pos['quantity']
                else:
                    pos['current_pnl'] = (pos['entry'] - current_price) * pos['quantity']

        return jsonify({
            'success': True,
            'positions': positions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions/<order_id>/close', methods=['POST'])
def close_position(order_id):
    """Close an active position"""
    try:
        if order_id not in state.active_positions:
            return jsonify({'success': False, 'error': 'Position not found'}), 404

        pos = state.active_positions[order_id]

        # Get current price
        market_data = get_cached_nifty_data() if pos['index'] == 'NIFTY' else get_cached_sensex_data()
        exit_price = market_data['spot_price']

        # Calculate P&L
        if pos['direction'] == 'CALL':
            pnl = (exit_price - pos['entry']) * pos['quantity']
        else:
            pnl = (pos['entry'] - exit_price) * pos['quantity']

        # Update position
        pos['status'] = 'closed'
        pos['exit_price'] = exit_price
        pos['pnl'] = pnl
        pos['exit_time'] = datetime.now().isoformat()

        # Send Telegram notification
        telegram = TelegramBot()
        telegram.send_position_closed(pos)

        return jsonify({
            'success': True,
            'position': pos,
            'message': f'Position closed with P&L: ₹{pnl:.2f}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - Bias Analysis
# ============================================================================

@app.route('/api/bias-analysis', methods=['GET'])
def get_bias_analysis():
    """Get comprehensive bias analysis"""
    try:
        # Lazy load bias analyzer
        if not state.bias_analyzer:
            state.bias_analyzer = BiasAnalysisPro()

        # Get cached results or run new analysis
        results = get_cached_bias_analysis_results()

        if not results:
            nifty_data = get_cached_nifty_data()
            results = state.bias_analyzer.analyze(nifty_data)

        return jsonify({
            'success': True,
            'bias_analysis': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - Chart Data
# ============================================================================

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    """Get historical chart data for technical analysis"""
    try:
        symbol = request.args.get('symbol', '^NSEI')  # Default to NIFTY
        period = request.args.get('period', '1d')
        interval = request.args.get('interval', '5m')

        # Lazy load chart analyzer
        if not state.chart_analyzer:
            state.chart_analyzer = AdvancedChartAnalysis()

        # Get chart data
        df = state.chart_analyzer.get_chart_data(symbol, period, interval)

        if df is None or len(df) == 0:
            return jsonify({'success': False, 'error': 'No data available'}), 404

        # Convert dataframe to JSON-friendly format
        chart_data = {
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': df['Open'].tolist(),
            'high': df['High'].tolist(),
            'low': df['Low'].tolist(),
            'close': df['Close'].tolist(),
            'volume': df['Volume'].tolist() if 'Volume' in df.columns else []
        }

        return jsonify({
            'success': True,
            'chart_data': chart_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/indicators', methods=['POST'])
def calculate_indicators():
    """Calculate technical indicators on chart data"""
    try:
        data = request.json
        indicator = data.get('indicator')
        params = data.get('params', {})

        # This would calculate indicators using chart_analyzer
        # Implementation depends on which indicators you want to support

        return jsonify({
            'success': True,
            'indicator_data': {}  # Return calculated indicator data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API ROUTES - AI Analysis
# ============================================================================

@app.route('/api/ai-analysis/run', methods=['POST'])
def run_ai_market_analysis():
    """Trigger AI market analysis"""
    try:
        # Get overall market sentiment
        overall_market = "NEUTRAL"
        if state.cached_sentiment:
            sentiment_map = {
                'BULLISH': 'BULL',
                'BEARISH': 'BEAR',
                'NEUTRAL': 'NEUTRAL'
            }
            overall_market = sentiment_map.get(
                state.cached_sentiment.get('overall_sentiment', 'NEUTRAL'),
                'NEUTRAL'
            )

        # Module biases (simplified)
        module_biases = {
            "htf_sr": 0.5,
            "vob": 0.5,
            "overall_sentiment": 0.5,
            "option_chain": 0.5,
            "proximity_alerts": 0.5,
        }

        # Market metadata
        nifty_data = get_cached_nifty_data()
        market_meta = {
            "volatility": 0.15,
            "volume_change": 0.05,
            "query": "NSE India market",
            "current_price": nifty_data.get('spot_price', 0) if nifty_data else 0,
            "market_status": get_market_status().get('session', 'unknown')
        }

        # Run AI analysis in background thread
        def run_async_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            report = loop.run_until_complete(
                run_ai_analysis(
                    overall_market,
                    module_biases,
                    market_meta,
                    save_report=True,
                    telegram_send=True
                )
            )
            loop.close()
            return report

        # Run in thread
        thread = threading.Thread(target=run_async_analysis, daemon=True)
        thread.start()

        state.last_ai_analysis_time = time.time()

        return jsonify({
            'success': True,
            'message': 'AI analysis started',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# WebSocket Events (Real-time updates)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to Flask server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('subscribe_market_data')
def handle_subscribe(data):
    """Subscribe to market data updates"""
    # In production, this would start pushing real-time updates
    # For now, we can emit periodic updates
    emit('market_data_update', {
        'nifty': get_cached_nifty_data(),
        'sensex': get_cached_sensex_data()
    })

# Background task to push real-time updates
def background_market_updates():
    """Push market updates to connected clients"""
    while True:
        time.sleep(5)  # Update every 5 seconds
        try:
            market_data = {
                'nifty': get_cached_nifty_data(),
                'sensex': get_cached_sensex_data(),
                'timestamp': datetime.now().isoformat()
            }
            socketio.emit('market_update', market_data, broadcast=True)
        except Exception as e:
            print(f"Error in background updates: {e}")

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 NIFTY/SENSEX Trading Dashboard - Flask Backend Server")
    print("=" * 80)
    print(f"Server starting on http://localhost:5000")
    print(f"Frontend will be available at: http://localhost:5000/")
    print(f"API endpoints available at: http://localhost:5000/api/*")
    print("=" * 80)

    # Start background market updates in a thread
    update_thread = threading.Thread(target=background_market_updates, daemon=True)
    update_thread.start()

    # Run Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
