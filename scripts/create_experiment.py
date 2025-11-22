"""Create a new experiment with proper versioning and documentation."""

import sys
import os
from datetime import datetime
import json
import yaml
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_next_version(strategy_type, description):
    """Determine next version number.

    Args:
        strategy_type: 'mean_reversion', 'trend_following', or 'combined'
        description: Brief description for version name

    Returns:
        Version string like 'v1.2_description'
    """
    exp_dir = f'experiments/{strategy_type}'

    if not os.path.exists(exp_dir):
        return f'v1.0_{description}'

    # Find existing versions
    versions = []
    for item in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, item)) and item.startswith('v'):
            try:
                ver_part = item.split('_')[0]  # e.g., 'v1.2'
                major, minor = ver_part[1:].split('.')
                versions.append((int(major), int(minor)))
            except:
                continue

    if not versions:
        return f'v1.0_{description}'

    # Get latest version
    latest = max(versions)
    next_minor = latest[1] + 1

    return f'v{latest[0]}.{next_minor}_{description}'


def create_experiment_dir(strategy_type, version, config_file=None):
    """Create experiment directory structure.

    Args:
        strategy_type: Strategy type
        version: Version string
        config_file: Path to config file to copy (optional)

    Returns:
        Path to experiment directory
    """
    exp_path = f'experiments/{strategy_type}/{version}'
    os.makedirs(exp_path, exist_ok=True)

    # Copy config if provided
    if config_file and os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(exp_path, 'config.yaml'))
        print(f"✓ Copied config from {config_file}")

    return exp_path


def create_summary_template(exp_path, version, strategy_type, changes_from_previous, rationale):
    """Create summary.md template.

    Args:
        exp_path: Path to experiment directory
        version: Version string
        strategy_type: Strategy type
        changes_from_previous: List of changes
        rationale: Why these changes were made
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    template = f"""# {strategy_type.replace('_', ' ').title()} - {version}

**Date**: {timestamp}
**Status**: 🔄 Running

---

## Changes from Previous Version

{chr(10).join('- ' + change for change in changes_from_previous)}

## Rationale

{rationale}

---

## Configuration

See `config.yaml` in this directory for complete configuration.

### Key Parameters:
```yaml
# Add key parameters here after running backtest
```

---

## Results

### Performance Metrics
```
Total Return:       N/A (running...)
CAGR:              N/A
Sharpe Ratio:      N/A
Max Drawdown:      N/A
Win Rate:          N/A
Total Trades:      N/A
Profit Factor:     N/A
```

### Comparison to Previous Version
```
Metric              Previous    Current     Change
-------------------------------------------------
Total Return
Sharpe Ratio
Total Trades
Win Rate
```

---

## Key Findings

(To be filled after backtest completes)

### What Worked:
-

### What Didn't Work:
-

### Unexpected Observations:
-

---

## Next Steps

(To be filled based on results)

1.
2.
3.

---

## Files in This Experiment

- `config.yaml` - Complete configuration
- `results.json` - Machine-readable metrics
- `summary.md` - This file
- `equity_curve.png` - Equity chart
- `drawdown.png` - Drawdown chart
- `trades.csv` - Complete trade log
- `equity_curve.csv` - Time series data

---

## Notes

"""

    summary_path = os.path.join(exp_path, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write(template)

    print(f"✓ Created summary template: {summary_path}")
    return summary_path


def save_results(exp_path, result):
    """Save backtest results to experiment directory.

    Args:
        exp_path: Experiment directory
        result: BacktestResult object
    """
    # Save metrics as JSON
    metrics_dict = result.metrics.to_dict()
    metrics_dict['symbol'] = result.symbol
    metrics_dict['timeframe'] = result.timeframe
    metrics_dict['start_date'] = result.start_date.isoformat()
    metrics_dict['end_date'] = result.end_date.isoformat()
    metrics_dict['initial_capital'] = result.initial_capital
    metrics_dict['final_equity'] = result.final_equity

    results_path = os.path.join(exp_path, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"✓ Saved metrics: {results_path}")

    # Save trades CSV
    import pandas as pd
    trades_df = pd.DataFrame([{
        'timestamp': t.timestamp,
        'symbol': t.symbol,
        'side': t.side,
        'quantity': t.quantity,
        'price': t.price,
        'fee': t.fee,
        'realized_pnl': t.realized_pnl,
        'strategy': t.strategy_name
    } for t in result.trades])

    trades_path = os.path.join(exp_path, 'trades.csv')
    trades_df.to_csv(trades_path, index=False)
    print(f"✓ Saved trades: {trades_path}")

    # Save equity curve
    equity_df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
    equity_path = os.path.join(exp_path, 'equity_curve.csv')
    equity_df.to_csv(equity_path, index=False)
    print(f"✓ Saved equity curve: {equity_path}")


def main():
    """Interactive experiment creation."""
    print("="*60)
    print("EXPERIMENT CREATION WIZARD")
    print("="*60)

    # Get strategy type
    print("\nStrategy type:")
    print("  1. Mean Reversion")
    print("  2. Trend Following")
    print("  3. Combined")

    choice = input("\nSelect (1-3): ").strip()
    strategy_map = {
        '1': 'mean_reversion',
        '2': 'trend_following',
        '3': 'combined'
    }

    strategy_type = strategy_map.get(choice)
    if not strategy_type:
        print("Invalid choice")
        return 1

    # Get description
    description = input("\nBrief description (snake_case, e.g., 'market_regime_filter'): ").strip()
    if not description:
        print("Description required")
        return 1

    # Get version
    version = get_next_version(strategy_type, description)
    print(f"\n→ Next version: {version}")

    # Get changes
    print("\nList changes from previous version (one per line, empty line to finish):")
    changes = []
    while True:
        change = input("  - ").strip()
        if not change:
            break
        changes.append(change)

    if not changes:
        print("At least one change required")
        return 1

    # Get rationale
    rationale = input("\nRationale for these changes: ").strip()
    if not rationale:
        print("Rationale required")
        return 1

    # Create experiment
    print(f"\n{'='*60}")
    print(f"Creating experiment: {strategy_type}/{version}")
    print(f"{'='*60}\n")

    exp_path = create_experiment_dir(strategy_type, version, 'config/default.yaml')
    print(f"✓ Created directory: {exp_path}")

    create_summary_template(exp_path, version, strategy_type, changes, rationale)

    print(f"\n{'='*60}")
    print("EXPERIMENT CREATED!")
    print(f"{'='*60}\n")
    print(f"Directory: {exp_path}")
    print(f"\nNext steps:")
    print(f"1. Edit {exp_path}/config.yaml if needed")
    print(f"2. Run your backtest")
    print(f"3. Use save_results() to save results to this directory")
    print(f"4. Update {exp_path}/summary.md with findings")
    print(f"5. Update experiments/CHANGELOG.md with entry")

    return 0


if __name__ == '__main__':
    sys.exit(main())
