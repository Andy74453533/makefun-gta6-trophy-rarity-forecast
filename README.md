[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Andy74453533/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Andy74453533/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecast how many GTA VI players earn each trophy in week one — an entry for the [$250 bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

For each of the 55 official GTA VI PS5 trophies, the program outputs five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate (percent of tracked players) at the case's target time, given the Exophase read history up to the case's checkpoint.

## How it works

The earn rate is `earners / players_tracked`, and the two sides move for different reasons. The model treats them separately:

- **Earners — cohort-delay model.** Every player cohort that joins at time `s` earns trophy `i` after a lognormal delay `D_i` (median `m_i`, σ = 0.9). The earners at time `t` are `F_i × Σ_k ΔN_k × P(D_i ≤ t − s_k)`, where `F_i` is the fraction of players who ever earn the trophy. `m_i` is fitted per trophy on a grid (2h–512h) by least squares against the observed history; `F_i` is refit at the shrunk delay so the forecast stays anchored to the observed level. Trophies that barely moved fall back toward class priors (story chapters, mission trophies, grinds, gold/silver/bronze defaults) for the delay shape only — never for the level.
- **Tracked — power-law growth.** `players_tracked` grows roughly as `(1 + t)^b` during launch week; `b` is fitted by least squares on `log N` vs `log(1 + t)` (prior `b = 1` when the history is flat), with a prediction-interval spread. A staleness boost handles the case where the page has not refreshed for hours: the next refresh can jump the tracked base several-fold (launch night: 638 → 3,868 players in one refresh), so the forecast allows for that dilution.
- **Near-universal trophies** (last observed rate ≥ 90%, e.g. "Complete Gone Fishing" at 100%) are forecast from the observed rate — new players earn them within the hour, so dilution is tiny.
- **Spreads** combine the fit residual, the tracked-base uncertainty, a 2-decimal rounding term and a refresh-shock term that decays with the age of the last read, so the 90% band is honest about launch-week refresh jumps.

## Usage

```sh
python3 main.py
```

Reads one case from `$MAKEFUN_INPUT_DIR/<id>.in` (or `/input/<id>.in`) and writes `$MAKEFUN_OUTPUT_DIR/<id>.out` (or `/output/<id>.out`). Python 3.13, standard library only, no network at runtime, deterministic (two runs produce byte-identical output), ~0.06 s per case.

## Validation

- **Practice cases** (p00/p01, scored against synthetic answers built from the known 04:30Z refresh to 3,868 players tracked, earners growth 0–100%): pinball loss **0.79–0.93** vs the starter persistence baseline's **1.00–1.75**; typical median miss 0.22–0.27 pp vs 0.58–0.74 pp.
- **Synthetic final-like case** (60h history, 337 reads with 3-hour refresh steps, power-law tracked growth, known per-trophy delay/fraction ground truth, 4.5-day horizon): pinball loss **0.25** vs the starter's **0.73**; typical median miss 0.81 pp vs 2.63 pp; 90% coverage 0.91.
- Determinism: two runs on the kit's `input/` produce byte-identical outputs. Runtime: ~0.06 s per case (limit 8 s).

## Files

- `main.py` — the forecaster (stdlib only).
- `AGENT.md` — build notes for agents.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE`.
