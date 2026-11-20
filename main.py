"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
at the case's target, as five quantiles (5/25/50/75/95%), scored by pinball loss
on the logit scale.

Model: earners / players-tracked decomposition with a cohort-delay fit.
  rate_i(t) = E_i(t) / N(t).
  A player who starts the game earns trophy i after a random delay D_i
  (lognormal, median m_i). So E_i(t) = F_i * M_i(t), where M_i(t) is the
  number of tracked players old enough to have earned it:
      M_i(t) = sum over cohorts k of dN_k * P(D_i <= t - s_k),
  dN_k the players who joined at time s_k. F_i is the fraction of players
  who ever earn the trophy. Per trophy, (m_i, F_i) are fit by least squares
  over a grid of m_i, shrunk toward priors from the trophy's tier and
  description when the trophy barely moved. N(t) itself is fit as a power law
  in time since launch, blended with a launch-growth prior when the history
  is too flat to identify it; future cohorts (which dilute the rate, since
  they add to N but not yet to M) come from that N model. The spread is
  honest: fit residual + N-extrapolation uncertainty + answer-rounding noise
  + a refresh-shock term that decays as the history lengthens (Exophase
  refreshes in steps; a refresh can slash rates by dilution when the tracked
  base jumps).

Python 3.13 standard library only. Deterministic: the same case always
writes the same bytes. No network at runtime.

  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Grid of median delays (hours from starting the game to earning a trophy).
M_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512]
# Lognormal spread of the delay (log hours).
D_SIGMA = 0.9
# Power-law prior for N(t) growth when the history is too flat to fit it.
N_PRIOR_B = 1.0
SPREAD_FLOOR = 0.35
REFRESH_SHOCK = 0.8
# Future cohorts are generated from the N model in steps of this many hours.
COHORT_STEP_H = 6.0


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delay_cdf(u, median):
    """P(delay <= u hours) for a lognormal delay with the given median."""
    if u <= 0:
        return 0.0
    return phi((math.log(u) - math.log(median)) / D_SIGMA)


def _classify(trophy, index, total):
    """Prior (median delay hours, eventual fraction) from trophy metadata.

    Story trophies run in list order (chapters 1-4 then the final mission,
    mission strands between them); activities and grinds sit further out
    with lower eventual fractions. Delays are CALENDAR hours from starting
    the game, not playtime: players play a few hours a day, so a chapter
    that takes 3 hours to play is earned a day or two after starting.
    """
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tier = (trophy.get("tier") or "bronze").lower()
    text = name + " " + desc
    if tier == "platinum":
        return (400.0, 0.002)
    if "gone fishing" in text:
        return (1.0, 0.99)
    if "chapter" in text or "final mission" in text:
        frac = index / max(1, total - 1)
        return (12.0 + 180.0 * frac, 0.55 - 0.45 * frac)
    if "complete" in text and ("mission" in text or "strand" in text or "run" in text or "heist" in text or "play" in text or "takedown" in text or "operation" in text):
        return (48.0, 0.25)
    if any(k in text for k in ("100% completion", "250 stunts", "gold medal", "50 challenges", "100 locations", "collectible card", "100 vehicles", "every spot", "garage propert", "laundering", "$250,000")):
        return (300.0, 0.008)
    if tier == "gold":
        return (150.0, 0.03)
    if tier == "silver":
        return (96.0, 0.06)
    return (48.0, 0.10)


def _fit_power_law(hours, log_n):
    """Least squares log N = a + b * log(1 + h). Returns (a, b)."""
    xs = [math.log1p(h) for h in hours]
    ys = list(log_n)
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    if not history:
        raise ValueError(f"{name} has no history")
    t0 = parse_time(history[0]["slot"])
    t_at = parse_time(case["at"])
    t_tgt = parse_time(case["target"])
    h_at = hours_between(t0, t_at)
    h_tgt = hours_between(t0, t_tgt)
    if h_tgt <= h_at:
        h_tgt = h_at + horizon_h

    # --- tracked-players series (monotone) --------------------------------
    hs = []
    ns = []
    run_max = 0.0
    for r in history:
        n = r.get("players_tracked")
        if n is None or n <= 0:
            continue
        run_max = max(run_max, float(n))
        hs.append(hours_between(t0, parse_time(r["slot"])))
        ns.append(run_max)
    if not hs:
        raise ValueError(f"{name} has no players_tracked")
    log_ns = [math.log(n) for n in ns]
    h_last = hs[-1]
    n_last = ns[-1]

    # Power-law fit for N, blended with the launch-growth prior when the
    # history is too flat to identify the growth (weight by how much N moved).
    _a_fit, b_fit = _fit_power_law(hs, log_ns)
    moved = log_ns[-1] - log_ns[0]
    w = min(1.0, max(0.0, moved / 0.5))
    b_use = w * b_fit + (1 - w) * N_PRIOR_B
    b_use = min(3.0, max(0.0, b_use))
    a_use = math.log(n_last) - b_use * math.log1p(h_last)

    def n_model(h):
        if h <= 0:
            return math.exp(a_use)
        return math.exp(a_use + b_use * math.log1p(h))

    n_tgt_model = n_model(h_tgt)
    # Uncertainty of the N extrapolation: the power-law fit's residual
    # spread, widened by leverage at the target (standard prediction
    # interval on log N). (A recent-exponential second opinion was tried
    # and dropped: it over-extrapolates power-law growth, whose log-rate
    # decays as 1/(1+h).)
    _a2, _b2 = _fit_power_law(hs, log_ns)
    xs_pl = [math.log1p(h) for h in hs]
    mx_pl = sum(xs_pl) / len(xs_pl)
    sxx_pl = sum((x - mx_pl) ** 2 for x in xs_pl)
    if len(hs) > 2 and sxx_pl > 1e-12:
        s_res = math.sqrt(sum((y - (_a2 + _b2 * x)) ** 2 for x, y in zip(xs_pl, log_ns)) / (len(hs) - 2))
        lev = 1.0 + (math.log1p(h_tgt) - mx_pl) ** 2 / sxx_pl
        sigma_log_n = math.sqrt(s_res ** 2 * lev + 0.15 ** 2)
    else:
        sigma_log_n = 0.3
    sigma_log_n = min(max(sigma_log_n, 0.15), 0.8)
    n_tgt = n_tgt_model
    n_tgt = min(max(n_tgt, n_last * 1.05), n_last * 200.0)
    # Staleness: if N has not moved for hours, the page is overdue a refresh
    # and the next one can jump the tracked base (launch night: 638 -> 3868
    # in one refresh). Boost the target N by the stale stretch; in a live
    # history N moves every refresh, so this is ~1 there.
    last_change = hs[0]
    for j in range(1, len(ns)):
        if ns[j] != ns[j - 1]:
            last_change = hs[j]
    stale = h_last - last_change
    if stale > 1.0:
        n_tgt = max(n_tgt, n_last * math.exp(0.25 * min(stale, 6.0)))

    # Cohorts: (start hour, size). Observed cohorts from the N steps, then
    # future cohorts from the N model in COHORT_STEP_H steps to the target.
    cohorts = []
    prev = 0.0
    for h, n in zip(hs, ns):
        d = n - prev
        if d > 0:
            cohorts.append((h, d))
        prev = n
    h_step = h_last
    prev_n = n_last
    while h_step < h_tgt - 1e-9:
        h_next = min(h_tgt, h_step + COHORT_STEP_H)
        frac = (h_next - h_last) / max(1e-9, h_tgt - h_last)
        n_next = math.exp((1 - frac) * math.log(n_last) + frac * math.log(n_tgt))
        d = n_next - prev_n
        if d > 0:
            cohorts.append((h_next, d))
        prev_n = n_next
        h_step = h_next
    # Mature-player counts at each read, for every grid median delay.
    # M[m][j] = players old enough (delay <= h_j - s_k) by read j.
    mature = {}
    for m in M_GRID:
        col = []
        for h in hs:
            tot = 0.0
            for (s, d) in cohorts:
                if s > h:
                    break
                tot += d * delay_cdf(h - s, m)
            col.append(tot)
        mature[m] = col

    def mature_at_target(median):
        tot = 0.0
        for (s, d) in cohorts:
            if s > h_tgt:
                break
            tot += d * delay_cdf(h_tgt - s, median)
        return tot

    # --- per-trophy fits ---------------------------------------------------
    rows = []
    total = len(trophies)
    for i, trophy in enumerate(trophies):
        m_prior, f_prior = _classify(trophy, i, total)
        # Align observed rates with the (players_tracked) reads.
        aligned = []
        for r in history:
            if r.get("players_tracked") is None or r.get("players_tracked") <= 0:
                continue
            rates = r.get("rates") or []
            aligned.append(None if i >= len(rates) or rates[i] is None else rates[i] / 100.0)
        ys = aligned
        obs = [(j, y) for j, y in enumerate(ys) if y is not None]
        if not obs:
            med = f_prior * mature_at_target(m_prior) / n_tgt
            rows.append((trophy["id"], med, 1.2))
            continue
        # Near-universal trophies (nearly everyone earns them on starting):
        # the cohort model would dilute them wrongly. Trust the observed
        # rate; new players earn them within the hour, so dilution is tiny.
        if obs[-1][1] >= 0.90:
            rows.append((trophy["id"], min(0.999, obs[-1][1]), 0.25))
            continue
        lg = [logit(max(0.01, min(99.99, y * 100.0))) for _j, y in obs]
        d_logit = (max(lg) - min(lg)) if lg else 0.0
        shrink = 1.0 / (1.0 + (d_logit / 0.7) ** 2)
        best = None
        for m in M_GRID:
            col = mature[m]
            sxx = 0.0
            sxy = 0.0
            for j, y in obs:
                x = col[j] / ns[j] if ns[j] > 0 else 0.0
                sxx += x * x
                sxy += x * y
            if sxx <= 1e-12:
                continue
            f_fit = min(1.0, max(0.0, sxy / sxx))
            ss = 0.0
            for j, y in obs:
                x = col[j] / ns[j] if ns[j] > 0 else 0.0
                ss += (y - f_fit * x) ** 2
            if best is None or ss < best[0]:
                best = (ss, m, f_fit)
        if best is None:
            med = f_prior * mature_at_target(m_prior) / n_tgt
            rows.append((trophy["id"], med, 1.2))
            continue
        ss, m_fit, f_fit = best
        # Shrink only the SHAPE (median delay) toward the prior when the
        # trophy barely moved; the LEVEL is observed, so refit F at the
        # shrunk delay instead of shrinking F to the prior.
        m_use = math.exp((1 - shrink) * math.log(m_fit) + shrink * math.log(m_prior))
        col_use = []
        for h in hs:
            tot = 0.0
            for (s, d) in cohorts:
                if s > h:
                    break
                tot += d * delay_cdf(h - s, m_use)
            col_use.append(tot)
        sxx = 0.0
        sxy = 0.0
        for j, y in obs:
            x = col_use[j] / ns[j] if ns[j] > 0 else 0.0
            sxx += x * x
            sxy += x * y
        f_use = min(1.0, max(0.0, sxy / sxx)) if sxx > 1e-12 else f_prior
        # Residual spread in logit units, at (m_use, f_use).
        res = []
        for j, y in obs:
            pred = f_use * (col_use[j] / ns[j] if ns[j] > 0 else 0.0)
            res.append(logit(max(0.01, min(99.99, y * 100.0))) - logit(max(0.01, min(99.99, pred * 100.0))))
        if res:
            mean_res = sum(res) / len(res)
            s_fit = math.sqrt(sum((x - mean_res) ** 2 for x in res) / len(res))
        else:
            s_fit = 0.5
        s_fit = min(1.0, s_fit) * math.sqrt(1.0 + (h_tgt - h_last) / 48.0)
        # Degenerate fit: if the effective times carry fewer than 3 distinct
        # values (e.g. a flat history where every read shows one stale
        # snapshot), the residuals are ~0 and the spread would collapse to
        # overconfidence. Floor the fit spread in that case.
        if len({hs[j] for j, _ in obs}) < 3:
            s_fit = max(s_fit, 0.5)
        med = f_use * mature_at_target(m_use) / n_tgt
        med = min(99.99, max(0.005, med * 100.0)) / 100.0
        s_round = min(0.5, 0.005 / max(med * 100.0, 0.02))
        s_refresh = REFRESH_SHOCK * math.exp(-h_last / 24.0)
        spread = math.sqrt(SPREAD_FLOOR ** 2 + s_fit ** 2 + (1.5 * sigma_log_n) ** 2 + s_round ** 2 + s_refresh ** 2)
        spread = min(2.0, max(0.3, spread))
        rows.append((trophy["id"], med, spread))

    out = {"trophies": []}
    for tid, med, spread in rows:
        lg = logit(med * 100.0)
        qs = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out["trophies"].append({"id": tid, "q": qs})
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
