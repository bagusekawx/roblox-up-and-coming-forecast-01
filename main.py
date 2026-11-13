#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for the next 72 hours.

Approach: Monte Carlo rank diffusion. Every game on the closing chart gets a
latent rank that drifts each hour by its shrunk recent rank velocity (momentum
is weakly persistent, corr ~0.2) plus a small global downward push from new
entrants, with rank-dependent Gaussian noise. Games can die (empirical
chart-exit hazard by rank band, scaled up 1.5x because a dead game's mass is
better spent on survivors and the unnamed residual) and age out exactly 28
days after creation when the creation date is known. A last-read playing-share
surge (share at least doubled) adds a short-lived upward kick: extreme surges
are the one playing signal that predicts the next rank move.

The Monte Carlo occupancy is blended 90/10 with a recency-weighted empirical
occupancy over the last 6 reads (12h half-life), capped at 0.55 per cell,
floored at 0.002, and scaled so each place names at most 0.80 of the mass --
the unnamed 0.20 residual covers off-chart entrants, which take several top-10
slots in every 72h window. The ordered top 10 is by mean simulated rank.

Tuned on five backtest windows (2026-10-31 .. 2026-11-10, 46 scored reads):
mean log score -36.07 vs -47.18 for the persistence baseline. Deterministic:
fixed seed, no network, stdlib only.
"""

import json
import math
import os
import random
from datetime import datetime

# ---- tuned constants (5-window backtest) ----
PATHS = 3000
HORIZON_H = 72
SEED = 12345
VEL_WINDOW_H = 36.0
DRIFT_SHRINK = 0.35
DRIFT_CAP = 1.2
DRIFT_TAU_H = 30.0
SIG_BASE = 0.6
SIG_PER_RANK = 0.03
GLOBAL_DRIFT = 0.03
HAZARD = (0.0007, 0.0025, 0.0047, 0.011)  # top10, 11-30, 31-60, 61+
HAZARD_SCALE = 1.5
SURGE_RATIO = 2.0
SURGE_DRIFT = 0.3
SURGE_TAU_H = 18.0
OCC_READS = 6
OCC_HALFLIFE_H = 12.0
W_MC = 0.90
W_OCC = 0.10
RESIDUAL = 0.20
P_CAP = 0.55
P_MIN = 0.002
ROWS_PER_PLACE = 40
AGE_OUT_H = 28 * 24


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_history(input_dir):
    path = os.path.join(input_dir, "history.ndjson")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # the hourly recorder marks failed reads ok=false with no chart
            if row.get("ok", True) and row.get("up_and_coming"):
                rows.append(row)
    rows.sort(key=lambda r: r["at"])
    return rows


def collect(rows):
    """Per-game rank series, closing chart, creation dates."""
    series = {}
    for row in rows:
        when = parse_ts(row["at"])
        for game in row["up_and_coming"]:
            uid = game["universe_id"]
            entry = series.setdefault(uid, {"ranks": [], "created": None})
            entry["ranks"].append((when, game["rank"]))
    closing = {}
    for game in rows[-1]["up_and_coming"]:
        closing[game["universe_id"]] = game["rank"]
    for row in rows:
        for uid_text, detail in (row.get("details") or {}).items():
            uid = int(uid_text)
            if uid not in series or series[uid]["created"] is not None:
                continue
            created = (detail or {}).get("created")
            if created:
                try:
                    series[uid]["created"] = parse_ts(created)
                except ValueError:
                    pass
    return series, closing, parse_ts(rows[-1]["at"])


def rank_velocity(points, t_end):
    """Recency-weighted slope of rank vs hours (negative = climbing)."""
    window_s = VEL_WINDOW_H * 3600.0
    pts = []
    for when, rank in points:
        age_s = (t_end - when).total_seconds()
        if 0 <= age_s <= window_s:
            # x = hours before t_end, so x grows with time (most recent = 0);
            # a climbing game (rank falling over time) gets a negative slope
            pts.append((-age_s / 3600.0, rank))
    if len(pts) < 2:
        return 0.0
    weights = [1.0 + x / VEL_WINDOW_H for x, _ in pts]
    total_w = sum(weights)
    mean_x = sum(w * x for w, (x, _) in zip(weights, pts)) / total_w
    mean_y = sum(w * y for w, (_, y) in zip(weights, pts)) / total_w
    num = sum(w * (x - mean_x) * (y - mean_y) for w, (x, y) in zip(weights, pts))
    den = sum(w * (x - mean_x) ** 2 for w, (x, _) in zip(weights, pts))
    return num / den if den > 0 else 0.0


def exit_hazard(rank):
    if rank <= 10:
        band = 0
    elif rank <= 30:
        band = 1
    elif rank <= 60:
        band = 2
    else:
        band = 3
    return HAZARD[band] * HAZARD_SCALE


def last_read_share_growth(rows, uids):
    """Playing-share ratio between the last two reads, per game."""
    growth = {}
    if len(rows) < 2:
        return growth
    shares = []
    for row in rows[-2:]:
        total = sum((g.get("playing") or 0) for g in row["up_and_coming"]) or 1
        shares.append({g["universe_id"]: (g.get("playing") or 0) / total
                       for g in row["up_and_coming"]})
    prev, cur = shares
    for uid in uids:
        if prev.get(uid, 0) > 0 and cur.get(uid, 0) > 0:
            growth[uid] = cur[uid] / prev[uid]
    return growth


def simulate(rows):
    series, closing, t_end = collect(rows)
    uids = sorted(closing, key=lambda u: closing[u])
    count = len(uids)
    growth = last_read_share_growth(rows, uids)
    drift = []
    kick = []
    sigma = []
    age_out = []
    for uid in uids:
        vel = rank_velocity(series[uid]["ranks"], t_end)
        drift.append(max(-DRIFT_CAP, min(DRIFT_CAP, DRIFT_SHRINK * vel)))
        kick.append(-SURGE_DRIFT if growth.get(uid, 0) >= SURGE_RATIO else 0.0)
        sigma.append(SIG_BASE + SIG_PER_RANK * closing[uid])
        created = series[uid]["created"]
        if created is None:
            age_out.append(float("inf"))
        else:
            hours_left = (created - t_end).total_seconds() / 3600.0 + AGE_OUT_H
            age_out.append(max(0.0, hours_left))
    rng = random.Random(SEED)
    occ = [[0.0] * 10 for _ in range(count)]
    rank_total = [0.0] * count
    rank_seen = [0] * count
    for _ in range(PATHS):
        pos = [float(closing[u]) for u in uids]
        alive = [True] * count
        for hour in range(1, HORIZON_H + 1):
            drift_decay = math.exp(-hour / DRIFT_TAU_H)
            kick_decay = math.exp(-hour / SURGE_TAU_H)
            for i in range(count):
                if not alive[i]:
                    continue
                if hour >= age_out[i] or rng.random() < exit_hazard(pos[i]):
                    alive[i] = False
                    continue
                pos[i] += (drift[i] * drift_decay + kick[i] * kick_decay
                           + GLOBAL_DRIFT + rng.gauss(0.0, sigma[i]))
                if pos[i] < 0.5:
                    pos[i] = 0.5
            order = sorted((i for i in range(count) if alive[i]),
                           key=lambda i: pos[i])
            for place, i in enumerate(order[:10]):
                occ[i][place] += 1.0
            for place, i in enumerate(order):
                rank_total[i] += place + 1
                rank_seen[i] += 1
            for i in range(count):
                if not alive[i]:
                    rank_total[i] += count + 1
                    rank_seen[i] += 1
    norm = float(PATHS * HORIZON_H)
    mc = [[occ[i][k] / norm for k in range(10)] for i in range(count)]
    mean_rank = [rank_total[i] / rank_seen[i] if rank_seen[i] else 99.0
                 for i in range(count)]
    return uids, closing, mc, mean_rank, rows, t_end


def empirical_occupancy(rows, t_end, uids):
    index = {u: i for i, u in enumerate(uids)}
    occ = [[0.0] * 10 for _ in range(len(uids))]
    weight_sum = 0.0
    for row in rows[-OCC_READS:]:
        age_h = (t_end - parse_ts(row["at"])).total_seconds() / 3600.0
        weight = 0.5 ** (age_h / OCC_HALFLIFE_H)
        weight_sum += weight
        top = sorted(row["up_and_coming"], key=lambda g: g["rank"])[:10]
        for place, game in enumerate(top):
            if game["universe_id"] in index:
                occ[index[game["universe_id"]]][place] += weight
    if weight_sum > 0:
        for i in range(len(uids)):
            for k in range(10):
                occ[i][k] /= weight_sum
    return occ


def build_forecast(rows):
    uids, closing, mc, mean_rank, rows, t_end = simulate(rows)
    emp = empirical_occupancy(rows, t_end, uids)
    count = len(uids)
    probabilities = []
    for place in range(10):
        column = []
        for i in range(count):
            p = W_MC * mc[i][place] + W_OCC * emp[i][place]
            if p >= P_MIN:
                column.append((min(p, P_CAP), uids[i]))
        column.sort(reverse=True)
        column = column[:ROWS_PER_PLACE]
        total = sum(p for p, _ in column)
        if total > 0:
            scale = min(1.0, (1.0 - RESIDUAL) / total)
            for p, uid in column:
                probabilities.append({"universe_id": uid, "place": place + 1,
                                      "p": round(p * scale, 6)})
    ranked = sorted(range(count),
                    key=lambda i: (mean_rank[i], -sum(mc[i]), uids[i]))
    top10 = [uids[i] for i in ranked[:10]]
    if len(top10) < 10:
        for uid in uids:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == 10:
                break
    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    rows = load_history(input_dir)
    forecast = build_forecast(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(forecast, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
