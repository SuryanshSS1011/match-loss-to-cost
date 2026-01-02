# Changelog

## 2025-01-02 — Reviewer Response Fixes

### Fixed

- Capacity planning uses `max` instead of `percentile_95` from previous definition
- Oracle invariants now verified: zero overload, u_max ≤ 1/α
- SARIMA seasonality aligned to daily period (s=288)
- MAPE threshold centralized in config, unused epsilon parameter removed

### Added

- Test suite: 6 unit tests (golden snapshot, identity forecast), 14 integration tests
- Baseline models: Naive (persistence), Seasonal Naive, Holt-Winters
- Alpha sweep analysis with Pareto curve plots (`overload_vs_capacity.png`, `overload_vs_alpha.png`)
- Multi-seed experiment runner with mean±std uncertainty reporting
- sMAPE metric as alternative to threshold-masked MAPE
- GitHub Actions CI for automated unit testing
- Periodicity diagnostic function (`check_periodicity`)

### Changed

- LSTM window size: 24 → 72 (6-hour context for computational efficiency)
- SARIMA seasonal order: (1,0,1,72) → (1,0,1,288) (true daily periodicity)
- MAPE now reports per-link exclusion percentage
- Alpha sweep generates two separate plots for paper inclusion
- Test files renamed to descriptive names

### Infrastructure

- `pytest.ini` with integration test marker
- Auto-skip integration tests when artifacts missing
- Normalization stats saved to `results/normalization_stats.json` for reproducibility

### Project Structure

- Reorganized into `src/` (core modules), `scripts/` (entry points), `tests/`
- Moved `paper.tex` and `paper.pdf` to project root
- Updated all imports to use `src.*` package structure
