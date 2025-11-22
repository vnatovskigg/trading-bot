# Experiments Directory

This directory tracks all strategy optimization experiments with version control for configurations and complete performance history.

## Directory Structure

```
experiments/
├── README.md                          # This file
├── mean_reversion/                    # Mean reversion experiments
│   ├── v1.0_baseline/
│   │   ├── config.yaml               # Config snapshot
│   │   ├── results.json              # Metrics in JSON
│   │   ├── summary.md                # Human-readable summary
│   │   ├── equity_curve.png
│   │   ├── drawdown.png
│   │   └── trades.csv
│   ├── v1.1_reduced_frequency/
│   └── v1.2_extreme_thresholds/
├── trend_following/                   # Trend following experiments
│   ├── v1.0_baseline/
│   ├── v1.1_slower_ma/
│   └── v1.2_optimized/
├── combined/                          # Combined strategy experiments
│   └── v1.0_both_strategies/
└── CHANGELOG.md                       # Complete history of all experiments
```

## Naming Convention

Versions follow semantic versioning: `vX.Y_description`
- `X` = Major change (strategy logic change)
- `Y` = Minor change (parameter tuning)
- `description` = Brief identifier (snake_case)

## Experiment Metadata

Each experiment directory contains:
1. **config.yaml** - Exact configuration used
2. **results.json** - Complete metrics in machine-readable format
3. **summary.md** - Human-readable summary with:
   - Version information
   - Changes from previous version
   - Rationale for changes
   - Performance comparison
   - Key findings
   - Next steps
4. **Performance artifacts** - Charts, trade logs, equity curves

## Workflow

1. Create new version directory: `experiments/strategy_name/vX.Y_description/`
2. Copy config to version directory
3. Run backtest
4. Save all results to version directory
5. Create summary.md documenting changes and results
6. Update CHANGELOG.md with entry
7. If performance improved, consider updating `config/default.yaml`

## Quick Reference Commands

```bash
# List all experiments
ls -la experiments/mean_reversion/
ls -la experiments/trend_following/

# View latest experiment
cat experiments/mean_reversion/v*/summary.md | tail -100

# Compare two versions
diff experiments/mean_reversion/v1.0_baseline/config.yaml \
     experiments/mean_reversion/v1.1_reduced_frequency/config.yaml
```

## Best Practices

1. **Never modify past experiments** - They are historical records
2. **Document everything** - Future you will thank present you
3. **Test incrementally** - Change one thing at a time when possible
4. **Track failed experiments** - Negative results are valuable
5. **Version control** - Commit experiment directories to git
