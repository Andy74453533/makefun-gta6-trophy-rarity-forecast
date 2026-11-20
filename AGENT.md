# AGENT.md — GTA VI Trophy Rarity Forecaster

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01` ($250, deadline
2026-11-21 11:30 UTC). Forecast the earn rate of each of the 55 GTA VI PS5
trophies at the case's target, as five quantiles per trophy.

## Run

```sh
python3 main.py          # reads $MAKEFUN_INPUT_DIR/<id>.in, writes $MAKEFUN_OUTPUT_DIR/<id>.out
```

Python 3.13 standard library only. No network at runtime. Deterministic
(byte-identical across runs). ~0.06 s per case (limit 8 s, 1 CPU, 1024 MB).

## Model (see README.md for the full description)

- Earners vs tracked, separately. Earners: cohort-delay model — each cohort
  of players earns a trophy after a lognormal delay (median fitted per
  trophy on a 2h–512h grid, σ = 0.9); the eventual-earn fraction is refit
  at the shrunk delay so the forecast stays anchored to the observed level.
- Tracked: power-law growth `(1 + t)^b` fitted on log N, with a staleness
  boost for overdue refreshes (launch night jumped 638 → 3,868 players in
  one refresh).
- Near-universal trophies (last rate ≥ 90%) forecast from the observed rate.
- Spreads combine fit residual + tracked-base uncertainty + rounding +
  refresh shock, aiming for honest 90% coverage.

## Validation (synthetic, in /tmp/val — not in the repo)

- Practice p00/p01 vs synthetic answers (04:30Z refresh, 3,868 tracked):
  loss 0.79–0.93 vs starter 1.00–1.75 across earners-growth scenarios.
- Synthetic final-like case (60h history, known ground truth, 4.5-day
  horizon): loss 0.25 vs starter 0.73, coverage 0.91.

## Notes for the submitter

- This competition takes transcript BUNDLES, not sealed recordings — the
  server refuses `record` for it (HTTP 400). Bundle with the level-20 kit
  CLI and submit with `--bundle`.
- README.md carries the three live make.fun badges from the start.
- Repo: https://github.com/Andy74453533/makefun-gta6-trophy-rarity-forecast
