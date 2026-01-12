"""
Multi-Dimensional Adaptive Trend Momentum Strategy
Main entry point for the trading system.

Usage:
    python main.py backtest --symbols 600000,000001 --start 20230101 --end 20231231
    python main.py live --symbols 600000 --capital 100000
    python main.py optimize --symbols 600000 --start 20230101 --end 20231231
"""
import argparse
import sys
from datetime import datetime
from typing import List

from config.settings import get_config, set_config, StrategyConfig
from data.data_fetcher import AShareDataFetcher
from backtest.backtester import Backtester
from strategy.market_state import MarketStateAnalyzer
from strategy.trend_analyzer import TrendAnalyzer
from strategy.signal_generator import SignalGenerator, SignalType
from strategy.position_manager import PositionManager, PositionSide
from execution.order_manager import OrderManager, OrderSide, OrderType
from execution.risk_control import RiskController
from utils.logger import setup_logger, get_logger, TradeLogger
from utils.gpu_utils import is_gpu_available, get_gpu_info


def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol to 6 digits with leading zeros."""
    symbol = symbol.strip()
    if symbol.isdigit() and len(symbol) < 6:
        return symbol.zfill(6)
    return symbol


def run_backtest(args):
    """Run historical backtest."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("MULTI-DIMENSIONAL ADAPTIVE TREND MOMENTUM STRATEGY")
    logger.info("Backtest Mode")
    logger.info("=" * 60)

    # Parse symbols and normalize (add leading zeros if needed)
    symbols = [normalize_symbol(s) for s in args.symbols.split(',')]

    # Initialize backtester
    backtester = Backtester(use_gpu=not args.no_gpu)

    # Run backtest
    result = backtester.run(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        timeframe=args.timeframe
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nPerformance Metrics:")
    print(f"  Total Return:      {result.total_return:.2%}")
    print(f"  Annualized Return: {result.annualized_return:.2%}")
    print(f"  Max Drawdown:      {result.max_drawdown:.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:.2f}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Win Rate:          {result.win_rate:.2%}")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Avg Win:           {result.avg_win:.2f}")
    print(f"  Avg Loss:          {result.avg_loss:.2f}")
    print(f"  Avg Duration:      {result.avg_trade_duration:.1f} days")
    print("=" * 60)

    # Save equity curve
    if args.output:
        result.equity_curve.to_csv(f"{args.output}_equity.csv")
        print(f"\nEquity curve saved to {args.output}_equity.csv")

    return result


def run_live(args):
    """Run live/paper trading."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("MULTI-DIMENSIONAL ADAPTIVE TREND MOMENTUM STRATEGY")
    logger.info("Live Trading Mode (Paper Trading)")
    logger.info("=" * 60)

    # Parse symbols and normalize (add leading zeros if needed)
    symbols = [normalize_symbol(s) for s in args.symbols.split(',')]

    # Initialize components
    data_fetcher = AShareDataFetcher()
    market_analyzer = MarketStateAnalyzer()
    trend_analyzer = TrendAnalyzer()
    signal_generator = SignalGenerator()
    position_manager = PositionManager(args.capital)
    order_manager = OrderManager()
    risk_controller = RiskController(position_manager)
    trade_logger = TradeLogger()

    config = get_config()

    print(f"\nMonitoring symbols: {symbols}")
    print(f"Initial capital: {args.capital:,.0f}")
    print(f"Press Ctrl+C to stop\n")

    try:
        import time

        while True:
            risk_controller.reset_daily_tracking()

            for symbol in symbols:
                try:
                    # Fetch latest data
                    df = data_fetcher.get_historical_data(
                        symbol,
                        timeframe=args.timeframe,
                        start_date=None,  # Uses default (1 year)
                        end_date=None
                    )

                    if df.empty or len(df) < 60:
                        continue

                    # Analyze market state
                    market_state = market_analyzer.get_current_state(
                        df['high'], df['low'], df['close']
                    )

                    # Get trend
                    trend = trend_analyzer.get_current_trend(df['close'])

                    # Get signal
                    trend_df = trend_analyzer.analyze(df['close'])
                    signal = signal_generator.get_current_signal(
                        df['close'], df['high'], df['low'],
                        df['volume'], trend_df['trend_score']
                    )

                    # Current price
                    current_price = df['close'].iloc[-1]
                    current_atr = market_state.atr_value

                    # Log status
                    logger.info(
                        f"{symbol}: Price={current_price:.2f} "
                        f"State={market_state.state.value} "
                        f"Trend={trend.direction.value} "
                        f"Score={signal.total_score:.1f}"
                    )

                    # Check risk
                    risk_status = risk_controller.check_risk()
                    if not risk_status.trading_allowed:
                        logger.warning(f"Trading restricted: {risk_status.reason}")
                        continue

                    # Update existing positions
                    if symbol in position_manager.positions:
                        action = position_manager.update_position(
                            symbol, current_price, current_atr
                        )
                        if action:
                            logger.info(f"Position action: {action}")

                    # Check for new entry
                    elif signal.signal_type != SignalType.NO_SIGNAL:
                        if market_state.state.value == 'trending':
                            # Calculate position size
                            size = position_manager.calculate_position_size(
                                current_price, current_atr, signal.total_score
                            )

                            logger.info(
                                f"SIGNAL: {signal.signal_type.value} {symbol} "
                                f"Score={signal.total_score:.1f} "
                                f"Confidence={signal.confidence} "
                                f"Size={size.quantity} shares"
                            )

                            # In paper trading, we would execute here
                            # For now, just log the signal

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Wait before next update
            logger.info(f"Sleeping {args.interval} seconds...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopping live trading...")

        # Print final portfolio status
        portfolio = position_manager.get_portfolio_risk()
        print("\nFinal Portfolio Status:")
        print(f"  Capital: {portfolio['current_capital']:,.2f}")
        print(f"  Open Positions: {portfolio['open_positions']}")
        print(f"  Total Exposure: {portfolio['exposure_pct']:.1f}%")


def run_optimize(args):
    """Run parameter optimization."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("PARAMETER OPTIMIZATION")
    logger.info("=" * 60)

    # Parse symbols and normalize
    symbols = [normalize_symbol(s) for s in args.symbols.split(',')]

    # Define parameter grid
    # Total combinations: 3 x 3 x 3 x 3 x 3 = 243
    param_grid = {
        'adx_threshold': [20, 25, 30],
        'ema_fast': [8, 10, 12],
        'ema_mid': [25, 30, 35],
        'rsi_oversold': [30, 35, 40],
        'rsi_overbought': [60, 65, 70],
    }

    # Initialize backtester
    backtester = Backtester(use_gpu=not args.no_gpu)

    # Run optimization
    result = backtester.optimize_parameters(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        param_grid=param_grid,
        metric=args.metric
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest Parameters:")
    for param, value in result['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest {args.metric}: {result['best_metric']:.4f}")
    print("=" * 60)


def show_info():
    """Show system information."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # GPU info
    print("\nGPU Status:")
    if is_gpu_available():
        info = get_gpu_info()
        print(f"  Available: Yes")
        print(f"  Library: {info['library']}")
        if info.get('device'):
            print(f"  Device: {info['device']}")
    else:
        print("  Available: No")
        print("  Install cupy or numba[cuda] for GPU acceleration")

    # Strategy config
    config = get_config()
    print("\nStrategy Parameters:")
    print(f"  ADX Trend Threshold: {config.market_state.trend_threshold}")
    print(f"  EMA Periods: {config.trend.ema_fast}/{config.trend.ema_mid}/{config.trend.ema_slow}")
    print(f"  MACD: {config.signal.macd_fast}/{config.signal.macd_slow}/{config.signal.macd_signal}")
    print(f"  RSI Period: {config.signal.rsi_period}")
    print(f"  Risk per Trade: {config.position.risk_per_trade:.1%}")
    print(f"  Max Drawdown: {config.risk_control.max_drawdown:.1%}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Dimensional Adaptive Trend Momentum Strategy"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run historical backtest')
    bt_parser.add_argument('--symbols', required=True, help='Comma-separated stock codes')
    bt_parser.add_argument('--start', required=True, help='Start date (YYYYMMDD)')
    bt_parser.add_argument('--end', required=True, help='End date (YYYYMMDD)')
    bt_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    bt_parser.add_argument('--timeframe', default='1d', help='Timeframe (1d, 1h, etc)')
    bt_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    bt_parser.add_argument('--output', help='Output file prefix')

    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live/paper trading')
    live_parser.add_argument('--symbols', required=True, help='Comma-separated stock codes')
    live_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    live_parser.add_argument('--timeframe', default='1d', help='Timeframe')
    live_parser.add_argument('--interval', type=int, default=60, help='Update interval (seconds)')

    # Optimization command
    opt_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    opt_parser.add_argument('--symbols', required=True, help='Comma-separated stock codes')
    opt_parser.add_argument('--start', required=True, help='Start date (YYYYMMDD)')
    opt_parser.add_argument('--end', required=True, help='End date (YYYYMMDD)')
    opt_parser.add_argument('--metric', default='sharpe_ratio', help='Optimization metric')
    opt_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')

    # Info command
    subparsers.add_parser('info', help='Show system information')

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=20)  # INFO level

    if args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'live':
        run_live(args)
    elif args.command == 'optimize':
        run_optimize(args)
    elif args.command == 'info':
        show_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
