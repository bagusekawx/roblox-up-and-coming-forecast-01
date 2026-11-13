#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for the next 72 hours (v2).

Monte Carlo rank diffusion (v1 core): every game on the closing chart gets a
latent rank that drifts each hour by its shrunk recent rank velocity (momentum
is weakly persistent, corr ~0.2) plus a small global downward push from new
entrants, with rank-dependent Gaussian noise. Games can die (empirical
chart-exit hazard by rank band, scaled up 1.5x because a dead game's mass is
better spent on survivors and the unnamed residual) and age out exactly 28
days after creation when the creation date is known. A last-read playing-share
surge (share at least doubled) adds a short-lived upward kick: extreme surges
are the one playing signal that predicts the next rank move.

v2 adds an explicit entrant component. The Monte Carlo gives deep-rank games
almost no top-10 mass (a rank-50 game's occupancy peaks below 0.0015), yet
games ranked 21+ hold about 18% of top-10 places over a 72h window. The
entrant component hands each deep band its empirical share of each place
(measured over 47 forecast points in the history) and splits it inside the
band by an entry score: climbing rank velocity (climbing 21-40 games enter at
19.8% vs 2.9% for falling ones), newness (first seen under 48h ago enters
4-5x as often), and absence from top-trending (already-trending games rarely
surge into this chart). The component replaces 75% of the Monte Carlo mass
for games ranked 21+, where the diffusion is known to be blind.

The Monte Carlo occupancy is blended 90/10 with a recency-weighted empirical
occupancy over the last 6 reads (12h half-life), capped at 0.55 per cell,
floored at 0.002, and scaled so each place names at most 1 - RESIDUAL[place]
of the mass. The residual is per-place in v2: bottom places churn harder and
leave more occupants unnamed (place 7 leaves 39% unnamed, place 5 only 11%),
so place 7 keeps 0.30 unnamed while place 5 keeps 0.16. The ordered top 10
is by mean simulated rank.

Validated on five backtest windows (2026-10-31 .. 2026-11-10, 46 scored
reads), 3000 paths, shipped code: mean log score -35.74 vs -36.04 for v1 and
-47.18 for the persistence baseline; practice window -33.33 vs -33.88 for v1
(W1 +0.56, W2 +0.34, W4 +0.81, W5 +0.21, W3 -0.42 on an 8-read window where a
single deep-rank game held places 1 and 3). Deterministic: fixed seed, no
network, stdlib only.
"""

import json
import math
import os
import random
from datetime import datetime

# ---- tuned constants (5-window backtest, v1 core) ----
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
P_CAP = 0.55
P_MIN = 0.002
ROWS_PER_PLACE = 40
AGE_OUT_H = 28 * 24
# ---- v2: per-place unnamed residual (place 1 .. 10) ----
RESIDUAL = (0.22, 0.22, 0.23, 0.20, 0.16, 0.22, 0.30, 0.18, 0.18, 0.24)
# ---- v2: entrant component ----
ENT_W = 0.75
ENT_MIN_RANK = 21
ENT_VEL_A = 0.6
ENT_NEW_H = 48.0
ENT_NEW_MULT = 2.5
ENT_TT_MULT = 0.5
# Empirical share of each place's occupants coming from each deep band,
# measured over 47 forecast points in the history (n=477 reads per place).
ENT_SHARE_21_40 = (0.090, 0.109, 0.061, 0.021, 0.027,
                   0.042, 0.109, 0.088, 0.119, 0.096)
ENT_SHARE_41P = (0.059, 0.105, 0.138, 0.040, 0.080,
                 0.092, 0.126, 0.140, 0.080, 0.117)


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


def entrant_matrix(uids, closing, raw_vel, age_list, on_tt):
    """Per-game entry mass by place for deep-rank games.

    Each deep band gets its empirical share of each place; inside a band the
    share splits by entry score: climbing velocity, newness, and absence from
    top-trending. Games ranked above ENT_MIN_RANK get no entrant mass.
    """
    count = len(uids)
    ent = [[0.0] * 10 for _ in range(count)]
    scores = []
    for i in range(count):
        if closing[uids[i]] < ENT_MIN_RANK:
            scores.append(0.0)
            continue
        vel = max(-3.0, min(3.0, raw_vel[i]))
        score = math.exp(-ENT_VEL_A * vel)
        if age_list[i] < ENT_NEW_H:
            score *= ENT_NEW_MULT
        if on_tt[i]:
            score *= ENT_TT_MULT
        scores.append(score)
    bands = ((21, 40, ENT_SHARE_21_40), (41, 10 ** 9, ENT_SHARE_41P))
    for lo, hi, share in bands:
        members = [i for i in range(count) if lo <= closing[uids[i]] <= hi]
        total = sum(scores[i] for i in members)
        if total <= 0:
            continue
        for i in members:
            weight = scores[i] / total
            for k in range(10):
                ent[i][k] = weight * share[k]
    return ent


def simulate(rows):
    series, closing, t_end = collect(rows)
    uids = sorted(closing, key=lambda u: closing[u])
    count = len(uids)
    growth = last_read_share_growth(rows, uids)
    tt_close = {g["universe_id"] for g in (rows[-1].get("top_trending") or [])}
    drift = []
    kick = []
    sigma = []
    age_out = []
    raw_vel = []
    age_list = []
    on_tt = []
    for uid in uids:
        vel = rank_velocity(series[uid]["ranks"], t_end)
        raw_vel.append(vel)
        first_seen = series[uid]["ranks"][0][0]
        age_list.append((t_end - first_seen).total_seconds() / 3600.0)
        on_tt.append(1 if uid in tt_close else 0)
        drift.append(max(-DRIFT_CAP, min(DRIFT_CAP, DRIFT_SHRINK * vel)))
        kick.append(-SURGE_DRIFT if growth.get(uid, 0) >= SURGE_RATIO else 0.0)
        sigma.append(SIG_BASE + SIG_PER_RANK * closing[uid])
        created = series[uid]["created"]
        if created is None:
            age_out.append(float("inf"))
        else:
            hours_left = (created - t_end).total_seconds() / 3600.0 + AGE_OUT_H
            age_out.append(max(0.0, hours_left))
    ent = entrant_matrix(uids, closing, raw_vel, age_list, on_tt)
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
    return uids, closing, mc, mean_rank, rows, t_end, ent


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
    uids, closing, mc, mean_rank, rows, t_end, ent = simulate(rows)
    emp = empirical_occupancy(rows, t_end, uids)
    count = len(uids)
    probabilities = []
    for place in range(10):
        column = []
        for i in range(count):
            p = W_MC * mc[i][place] + W_OCC * emp[i][place]
            if closing[uids[i]] >= ENT_MIN_RANK:
                p = (1.0 - ENT_W) * p + ENT_W * ent[i][place]
            if p >= P_MIN:
                column.append((min(p, P_CAP), uids[i]))
        column.sort(reverse=True)
        column = column[:ROWS_PER_PLACE]
        total = sum(p for p, _ in column)
        if total > 0:
            scale = min(1.0, (1.0 - RESIDUAL[place]) / total)
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
