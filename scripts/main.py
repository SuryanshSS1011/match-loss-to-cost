"""
Main script to run the complete network traffic forecasting experiment.

This script orchestrates the full pipeline:
1. Data generation
2. SARIMA model training and prediction
3. LSTM model training and prediction
4. Capacity planning evaluation
"""

import os
import sys
import time
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR


def run_data_generation():
    """Run data generation step."""
    print("\n" + "#" * 60)
    print("# STEP 1: DATA GENERATION")
    print("#" * 60)

    from src.simulate_data import main as simulate_main
    simulate_main()


def run_sarima_training():
    """Run SARIMA model training."""
    print("\n" + "#" * 60)
    print("# STEP 2: SARIMA MODEL TRAINING")
    print("#" * 60)

    from src.train_arima import main as arima_main
    arima_main()


def run_lstm_training():
    """Run LSTM model training."""
    print("\n" + "#" * 60)
    print("# STEP 3: LSTM MODEL TRAINING")
    print("#" * 60)

    from src.train_lstm import main as lstm_main
    lstm_main()


def run_evaluation():
    """Run capacity planning evaluation."""
    print("\n" + "#" * 60)
    print("# STEP 4: CAPACITY PLANNING EVALUATION")
    print("#" * 60)

    from src.eval_capacity import main as eval_main
    eval_main()


def print_final_summary():
    """Print final summary with file locations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print("\nGenerated Files:")
    print(f"\n  Data:")
    print(f"    - {os.path.join(DATA_DIR, 'topology.npz')}")
    print(f"    - {os.path.join(DATA_DIR, 'routing_matrix.npz')}")
    print(f"    - {os.path.join(DATA_DIR, 'traffic_data.npz')}")

    print(f"\n  Model:")
    print(f"    - {os.path.join('models', 'lstm_forecaster.pt')}")

    print(f"\n  Results:")
    for f in ['sarima_predictions.npz', 'lstm_predictions.npz',
              'sarima_metrics.json', 'lstm_metrics.json',
              'capacity_planning.json', 'combined_results.json']:
        print(f"    - {os.path.join(RESULTS_DIR, f)}")

    print(f"\n  Plots:")
    if os.path.exists(PLOTS_DIR):
        for f in sorted(os.listdir(PLOTS_DIR)):
            if f.endswith('.png'):
                print(f"    - {os.path.join(PLOTS_DIR, f)}")

    print("\n" + "=" * 60)


def main():
    """Run the complete experiment pipeline."""
    parser = argparse.ArgumentParser(
        description='Network Traffic Forecasting & Capacity Planning Experiment'
    )
    parser.add_argument(
        '--skip-data', action='store_true',
        help='Skip data generation (use existing data)'
    )
    parser.add_argument(
        '--skip-sarima', action='store_true',
        help='Skip SARIMA training (use existing predictions)'
    )
    parser.add_argument(
        '--skip-lstm', action='store_true',
        help='Skip LSTM training (use existing predictions)'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Only run evaluation (assumes data and predictions exist)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NETWORK TRAFFIC FORECASTING & CAPACITY PLANNING EXPERIMENT")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Nodes: {CONFIG['num_nodes']}")
    print(f"  - Days: {CONFIG['days']}")
    print(f"  - Time step: {CONFIG['time_step_minutes']} min")
    print(f"  - Window size: {CONFIG['window_size']}")
    print(f"  - SARIMA order: {CONFIG['arima_order']}")
    print(f"  - SARIMA seasonal: {CONFIG['seasonal_order']}")
    print(f"  - LSTM hidden: {CONFIG['lstm_hidden_size']}")
    print(f"  - LSTM layers: {CONFIG['lstm_num_layers']}")

    total_start = time.time()

    # Step 1: Data Generation
    if not args.skip_data and not args.eval_only:
        start = time.time()
        run_data_generation()
        print(f"\n   Data generation took: {time.time() - start:.1f}s")
    else:
        print("\n[Skipping data generation]")

    # Step 2: SARIMA Training
    if not args.skip_sarima and not args.eval_only:
        start = time.time()
        run_sarima_training()
        print(f"\n   SARIMA training took: {time.time() - start:.1f}s")
    else:
        print("\n[Skipping SARIMA training]")

    # Step 3: LSTM Training
    if not args.skip_lstm and not args.eval_only:
        start = time.time()
        run_lstm_training()
        print(f"\n   LSTM training took: {time.time() - start:.1f}s")
    else:
        print("\n[Skipping LSTM training]")

    # Step 4: Evaluation
    start = time.time()
    run_evaluation()
    print(f"\n   Evaluation took: {time.time() - start:.1f}s")

    # Final summary
    print_final_summary()

    total_time = time.time() - total_start
    print(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
