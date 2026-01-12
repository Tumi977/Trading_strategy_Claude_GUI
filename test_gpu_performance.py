"""
Performance Test Script.
Compares GPU/Numba parallel vs CPU performance for indicator calculations.
"""
import numpy as np
import time
import sys

# Add project to path
sys.path.insert(0, '.')

from utils.gpu_utils import GPUAccelerator, is_gpu_available, is_parallel_available, get_gpu_info


def generate_test_data(n_time: int, n_stocks: int) -> dict:
    """Generate random OHLCV test data."""
    np.random.seed(42)

    # Generate realistic price data using random walk
    base_prices = np.random.uniform(10, 100, n_stocks)
    returns = np.random.randn(n_time, n_stocks) * 0.02

    close = np.zeros((n_time, n_stocks))
    close[0] = base_prices
    for t in range(1, n_time):
        close[t] = close[t-1] * (1 + returns[t])

    # Generate OHLV from close
    daily_range = np.abs(np.random.randn(n_time, n_stocks) * 0.01) + 0.005
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    open_price = close * (1 + np.random.randn(n_time, n_stocks) * 0.005)
    volume = np.random.uniform(1e6, 1e8, (n_time, n_stocks))

    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def benchmark_indicators(gpu: GPUAccelerator, data: dict, n_runs: int = 5) -> dict:
    """Benchmark indicator calculations."""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']

    results = {}

    # Test EMA
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = gpu.ema_gpu(close, 20)
        times.append(time.perf_counter() - start)
    results['EMA'] = np.mean(times[1:])  # Skip first run (warmup)

    # Test ATR
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = gpu.atr_gpu(high, low, close, 14)
        times.append(time.perf_counter() - start)
    results['ATR'] = np.mean(times[1:])

    # Test ADX
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = gpu.adx_gpu(high, low, close, 14)
        times.append(time.perf_counter() - start)
    results['ADX'] = np.mean(times[1:])

    # Test MACD
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = gpu.macd_gpu(close, 12, 26, 9)
        times.append(time.perf_counter() - start)
    results['MACD'] = np.mean(times[1:])

    # Test RSI
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = gpu.rsi_gpu(close, 14)
        times.append(time.perf_counter() - start)
    results['RSI'] = np.mean(times[1:])

    return results


def main():
    print("=" * 60)
    print("PARALLEL COMPUTING PERFORMANCE TEST")
    print("=" * 60)

    # Check availability
    print("\nSystem Information:")
    info = get_gpu_info()

    if info.get('gpu_available'):
        print(f"  GPU: Available ({info.get('gpu_library')})")
        if info.get('device') and info['device'].get('type') == 'GPU':
            print(f"  Compute Capability: {info['device'].get('compute_capability')}")
            print(f"  Memory: {info['device'].get('memory_total_gb', 0):.1f} GB total")
    else:
        print("  GPU: Not available")

    if info.get('numba_parallel'):
        import multiprocessing
        print(f"  Numba CPU Parallel: Available ({multiprocessing.cpu_count()} cores)")
    else:
        print("  Numba CPU Parallel: Not available")

    # Test configurations
    test_configs = [
        (500, 10),     # Small: 500 days, 10 stocks
        (1000, 50),    # Medium: 1000 days, 50 stocks
        (2000, 100),   # Large: 2000 days, 100 stocks
        (5000, 200),   # XL: 5000 days, 200 stocks
    ]

    print("\n" + "-" * 60)
    print("Benchmark Results (time in milliseconds)")
    print("-" * 60)

    for n_time, n_stocks in test_configs:
        print(f"\nData size: {n_time} bars x {n_stocks} stocks = {n_time * n_stocks:,} data points")

        # Generate test data
        data = generate_test_data(n_time, n_stocks)

        # Parallel benchmark (GPU or Numba CPU)
        if is_parallel_available():
            parallel = GPUAccelerator(use_gpu=True)
            parallel_results = benchmark_indicators(parallel, data)
            backend = parallel.library
            print(f"\n  {backend} Results:")
            for name, time_sec in parallel_results.items():
                print(f"    {name:8s}: {time_sec * 1000:8.2f} ms")
            parallel.free_memory()

        # Pure NumPy benchmark (single-threaded)
        # Temporarily disable parallel to test numpy
        import utils.gpu_utils as gpu_module
        orig_numba = gpu_module._NUMBA_PARALLEL_AVAILABLE
        gpu_module._NUMBA_PARALLEL_AVAILABLE = False

        numpy_acc = GPUAccelerator(use_gpu=False)
        numpy_results = benchmark_indicators(numpy_acc, data)
        print("\n  NumPy (single-thread) Results:")
        for name, time_sec in numpy_results.items():
            print(f"    {name:8s}: {time_sec * 1000:8.2f} ms")

        gpu_module._NUMBA_PARALLEL_AVAILABLE = orig_numba

        # Speedup comparison
        if is_parallel_available():
            print(f"\n  Speedup ({backend} vs NumPy):")
            for name in parallel_results:
                speedup = numpy_results[name] / parallel_results[name] if parallel_results[name] > 0 else 0
                print(f"    {name:8s}: {speedup:8.2f}x")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
