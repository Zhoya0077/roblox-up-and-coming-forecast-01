"""Roblox Up-and-Coming chart forecaster (Monte Carlo latent-score model).

Reads /input/history.ndjson + task.json, writes forecast.json with per-place
probabilities and a top10. Model: every game ever seen carries a latent
score; scores follow an Ornstein-Uhlenbeck process (calibrated from the
history's own rank-change curve), games exit by a rank/age hazard, new games
enter at a calibrated rate, and absent games can re-enter. The chart is
simulated hourly over the scored window; place occupancy averaged over all
slots and simulations gives the per-place probabilities. Newcomer mass is
reserved as unnamed share (their ids are unknowable). Deterministic: one
seeded RNG, fixed iteration orders.

Stdlib only, no network. Reads window/places from task.json, never hardcodes
dates. Handles null details/votes/genres (archive reads) gracefully.
"""
import heapq
import json
import math
import os
import random

SEED = 0xC0FFEE
N_SIMS = 1500
# OU jump mixture: with JUMP_P the hourly shock is JUMP_MULT times wider.
# Captures the chart's bursty reshuffles (rank changes are heavy-tailed).
JUMP_P = 0.08
JUMP_MULT = 3.0
# Rank-velocity drift: shrunk, decaying, applied to latent scores.
DRIFT_SHRINK = 0.6
DRIFT_TAU_H = 24.0
DRIFT_CAP = 0.5  # ranks/hour
# Exit-hazard age tilt (created age in days when known, else neutral).
AGE_TILT_START = 21.0
AGE_TILT_MAX = 2.0
# Minimum unnamed reserve per place (covers model error on newcomers).
MIN_UNNAMED = 0.02
# Scale applied to the empirical newcomer-share reserve (tuned in dev).
RESERVE_SCALE = 1.4


def newcomer_share_curve(reads):
    """Share of top-10 slots held by never-before-seen games, vs horizon.

    From all read pairs (i, j) with i late enough that the seen-set is
    reasonably complete. Returns sorted [(dt_hours, share)] bucket means.
    """
    n = len(reads)
    min_i = min(10, max(0, n // 3))
    buckets = {}
    for i in range(min_i, n):
        seen = set()
        for k in range(i):
            seen |= set(reads[k]["chart"].keys())
        for j in range(i + 1, n):
            dt = reads[j]["t"] - reads[i]["t"]
            if dt <= 0:
                continue
            b = max(6, int(round(dt / 12.0)) * 12)
            acc = buckets.setdefault(b, [0, 0])
            top = sorted(reads[j]["chart"], key=lambda u: reads[j]["chart"][u])[:10]
            for uid in top:
                acc[1] += 1
                if uid not in seen:
                    acc[0] += 1
    pts = sorted((b, a[0] / a[1]) for b, a in buckets.items() if a[1] >= 40)
    return pts


def newcomer_reserve(reads, horizons):
    """Mean never-seen share over the given horizons (hours after last read)."""
    pts = newcomer_share_curve(reads)
    if len(pts) < 3:
        return 0.08
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    total = 0.0
    for h in horizons:
        if h <= xs[0]:
            total += ys[0] * (h / xs[0])
        elif h >= xs[-1]:
            total += ys[-1]
        else:
            for a in range(len(xs) - 1):
                if xs[a] <= h <= xs[a + 1]:
                    f = (h - xs[a]) / (xs[a + 1] - xs[a])
                    total += ys[a] + f * (ys[a + 1] - ys[a])
                    break
    mean_share = total / len(horizons) if horizons else 0.08
    return min(max(mean_share * RESERVE_SCALE, MIN_UNNAMED), 0.30)


def parse_time(s):
    # ISO 8601 UTC -> epoch hours (float).
    return _iso_to_epoch(s) / 3600.0


def _iso_to_epoch(s):
    # Minimal ISO parser (stdlib, no datetime strptime cost concerns).
    # Accepts 'YYYY-MM-DDTHH:MM:SS(.fff)?Z'.
    from datetime import datetime, timezone

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def load_reads(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = {}
            for g in r.get("up_and_coming") or []:
                chart[g["universe_id"]] = g["rank"]
            created = {}
            details = r.get("details")
            if details:
                for uid_s, d in details.items():
                    try:
                        uid = int(uid_s)
                    except (TypeError, ValueError):
                        continue
                    if d and d.get("created"):
                        created[uid] = parse_time(d["created"])
            reads.append({"t": parse_time(r["at"]), "chart": chart, "created": created})
    reads.sort(key=lambda r: r["t"])
    return task, reads


def empirical_std_curve(reads):
    """Std of rank change vs lag (hours), from all read pairs, survivors only."""
    buckets = {}
    n = len(reads)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 4000:  # subsample evenly for speed on dense histories
        step = len(pairs) / 4000.0
        pairs = [pairs[int(k * step)] for k in range(4000)]
    for i, j in pairs:
        dt = reads[j]["t"] - reads[i]["t"]
        if dt <= 0:
            continue
        b = max(6, int(round(dt / 6.0)) * 6)
        acc = buckets.setdefault(b, [0, 0, 0])
        ci, cj = reads[i]["chart"], reads[j]["chart"]
        for uid, rk in ci.items():
            if uid in cj:
                d = cj[uid] - rk
                acc[0] += 1
                acc[1] += d
                acc[2] += d * d
    curve = {}
    for b, (cnt, s1, s2) in buckets.items():
        if cnt >= 30:
            mean = s1 / cnt
            var = max(s2 / cnt - mean * mean, 0.25)
            curve[b] = (math.sqrt(var), cnt)
    return curve


def fit_ou(curve):
    """Fit std(dt)^2 = sigma^2 * tau * (1 - exp(-2 dt / tau)) by grid over tau."""
    if len(curve) < 3:
        return 2.7, 160.0
    pts = sorted(curve.items())
    best = None
    tau = 30.0
    while tau <= 400.0:
        num = den = err = 0.0
        for dt, (std, cnt) in pts:
            f = tau * (1.0 - math.exp(-2.0 * dt / tau))
            w = math.sqrt(cnt)
            y = std * std
            num += w * y * f
            den += w * f * f
        sig2 = num / den if den > 0 else 7.3
        for dt, (std, cnt) in pts:
            pred = math.sqrt(sig2 * tau * (1.0 - math.exp(-2.0 * dt / tau)))
            err += math.sqrt(cnt) * (pred - std) ** 2
        if best is None or err < best[0]:
            best = (err, math.sqrt(max(sig2, 0.25)), tau)
        tau += 10.0
    _, sigma, tau = best
    return min(max(sigma, 0.5), 8.0), tau


def hazard_rates(reads):
    """Per-hour exit hazard by rank band + entry/re-entry stats, from history."""
    bands = [(1, 10), (11, 30), (31, 60), (61, 10 ** 9)]
    expo = [0.0] * 4
    exits = [0] * 4
    first_seen = {}
    entries = []
    reenter_events = 0
    absent_hours = 0.0
    ever = set()
    for idx, r in enumerate(reads):
        for uid in r["chart"]:
            if uid not in first_seen:
                first_seen[uid] = idx
                if idx > 0:
                    entries.append(r["chart"][uid])
            ever.add(uid)
    for i in range(len(reads) - 1):
        dt = reads[i + 1]["t"] - reads[i]["t"]
        if dt <= 0:
            continue
        ci, cj = reads[i]["chart"], reads[i + 1]["chart"]
        for uid, rk in ci.items():
            b = 0 if rk <= 10 else 1 if rk <= 30 else 2 if rk <= 60 else 3
            expo[b] += dt
            if uid not in cj:
                exits[b] += 1
        absent = ever - set(ci.keys())
        absent_hours += len(absent) * dt
        for uid in cj:
            if uid not in ci and uid in ever and first_seen.get(uid, 10 ** 9) < i:
                reenter_events += 1
    hazards = []
    for b in range(4):
        h = exits[b] / expo[b] if expo[b] > 0 else 0.002
        hazards.append(min(max(h, 0.0002), 0.05))
    total_h = reads[-1]["t"] - reads[0]["t"] if len(reads) > 1 else 1.0
    entry_rate = len(entries) / total_h if total_h > 0 else 0.5
    reenter_rate = reenter_events / absent_hours if absent_hours > 0 else 0.0
    return hazards, min(max(entry_rate, 0.05), 2.0), min(reenter_rate, 0.01), entries


def rank_velocities(reads, last_idx):
    """Shrunk recent rank velocity (ranks/hour, negative = improving) per game."""
    vel = {}
    if len(reads) < 3:
        return vel
    k0 = max(0, len(reads) - 5)
    for uid in reads[last_idx]["chart"]:
        pts = [(reads[i]["t"], reads[i]["chart"][uid]) for i in range(k0, len(reads)) if uid in reads[i]["chart"]]
        if len(pts) < 3:
            continue
        n = len(pts)
        mt = sum(p[0] for p in pts) / n
        mr = sum(p[1] for p in pts) / n
        den = sum((p[0] - mt) ** 2 for p in pts)
        if den <= 0:
            continue
        slope = sum((p[0] - mt) * (p[1] - mr) for p in pts) / den
        v = DRIFT_SHRINK * slope
        vel[uid] = max(-DRIFT_CAP, min(DRIFT_CAP, v))
    return vel


def simulate(task, reads, n_sims=N_SIMS, seed=SEED):
    places = int(task["places"])
    win_from = parse_time(task["window"]["from"])
    win_until = parse_time(task["window"]["until"])
    slots = int(task["window"]["slots"])
    slot_starts = [win_from + h * (win_until - win_from) / slots for h in range(slots)]

    last = reads[-1]
    t0 = last["t"]
    horizon_end = win_until
    n_steps = max(1, int(math.ceil(horizon_end - t0)))
    chart_sizes = [len(r["chart"]) for r in reads[-5:]]
    n_chart = int(sorted(chart_sizes)[len(chart_sizes) // 2])

    curve = empirical_std_curve(reads)
    sigma, tau = fit_ou(curve)
    # Jump mixture preserves total variance: scale base sigma accordingly.
    jump_var = (1.0 - JUMP_P) + JUMP_P * JUMP_MULT * JUMP_MULT
    base_sigma = sigma / math.sqrt(jump_var)
    hazards, entry_rate, reenter_rate, entry_ranks = hazard_rates(reads)
    if not entry_ranks:
        entry_ranks = [n_chart]
    vel = rank_velocities(reads, len(reads) - 1)

    # Created ages: newest known created per game across reads.
    created = {}
    for r in reads:
        for uid, ct in r["created"].items():
            created[uid] = ct

    all_uids = sorted({uid for r in reads for uid in r["chart"]})
    uid_index = {u: i for i, u in enumerate(all_uids)}
    last_chart = last["chart"]
    last_rank = {u: last_chart[u] for u in all_uids if u in last_chart}
    # Last-seen rank + time for absent games (for re-entry scoring).
    last_seen_rank = {}
    for r in reads:
        for uid, rk in r["chart"].items():
            last_seen_rank[uid] = rk

    mu = -(n_chart + 1) / 2.0  # OU mean: middle of the chart in score space
    decay = math.exp(-1.0 / tau)
    ou_noise = base_sigma * math.sqrt(max(tau * 0.5 * (1.0 - decay * decay), 1e-9))
    drift_decay = math.exp(-1.0 / DRIFT_TAU_H)

    occ = [[0] * places for _ in all_uids]  # occ[game][place-1]
    newcomer_occ = [0] * places
    total_slot_sims = 0

    rng = random.Random(seed)
    for _ in range(n_sims):
        # State arrays, indexed like all_uids; newcomers appended dynamically.
        score = [0.0] * len(all_uids)
        active = [False] * len(all_uids)
        for uid in all_uids:
            i = uid_index[uid]
            if uid in last_rank:
                score[i] = -float(last_rank[uid])
                active[i] = True
            else:
                score[i] = -(n_chart + 30.0)
        new_scores = []  # scores of newcomer games (aggregate only)
        new_active = []
        # Per-game exit-time override: none (hazard only; no hard age cliff).
        slot_ptr = 0
        # Precompute slot step indices (hour steps from t0).
        slot_steps = set()
        for s in slot_starts:
            st = int(round(s - t0))
            if 0 <= st <= n_steps:
                slot_steps.add(st)
        drift = [0.0] * len(all_uids)
        for uid, v in vel.items():
            drift[uid_index[uid]] = -v  # improving rank -> rising score
        for step in range(n_steps + 1):
            t = t0 + step
            if step in slot_steps:
                cands = [(score[i], i) for i in range(len(all_uids)) if active[i]]
                cands += [(s, -1 - k) for k, s in enumerate(new_scores) if new_active[k]]
                top = heapq.nlargest(places, cands)
                for pos, (_, gi) in enumerate(top):
                    if gi >= 0:
                        occ[gi][pos] += 1
                    else:
                        newcomer_occ[pos] += 1
                total_slot_sims += 1
            if step == n_steps:
                break
            # OU + drift + noise for active known games.
            for i in range(len(all_uids)):
                if not active[i]:
                    continue
                z = rng.gauss(0.0, 1.0)
                if rng.random() < JUMP_P:
                    z *= JUMP_MULT
                s = mu + (score[i] - mu) * decay + ou_noise * z + drift[i]
                score[i] = s
                drift[i] *= drift_decay
            # Newcomer dynamics (same OU, no drift).
            for k in range(len(new_scores)):
                if not new_active[k]:
                    continue
                z = rng.gauss(0.0, 1.0)
                if rng.random() < JUMP_P:
                    z *= JUMP_MULT
                new_scores[k] = mu + (new_scores[k] - mu) * decay + ou_noise * z
            # Recompute current ranks for hazard bands (rank from position
            # in the score sort).
            ranked = sorted(((score[i], i) for i in range(len(all_uids)) if active[i]), reverse=True)
            rank_of = {}
            for pos, (_, i) in enumerate(ranked):
                rank_of[i] = pos + 1
            # Exits.
            for i in range(len(all_uids)):
                if not active[i]:
                    continue
                rk = rank_of.get(i, n_chart + 1)
                b = 0 if rk <= 10 else 1 if rk <= 30 else 2 if rk <= 60 else 3
                h = hazards[b]
                uid = all_uids[i]
                if uid in created:
                    age_d = (t - created[uid]) / 24.0
                    if age_d > AGE_TILT_START:
                        h *= min(AGE_TILT_MAX, 1.0 + 0.8 * (age_d - AGE_TILT_START) / 14.0)
                if rng.random() < h:
                    active[i] = False
            # Newcomer exits (same bottom-band-ish hazard by their rank).
            for k in range(len(new_scores)):
                if new_active[k] and rng.random() < hazards[3]:
                    new_active[k] = False
            # Entries: Poisson(entry_rate) newcomers this hour.
            lam = entry_rate
            L = math.exp(-lam)
            k_entries = 0
            p = 1.0
            while True:
                p *= rng.random()
                if p <= L:
                    break
                k_entries += 1
            for _ in range(k_entries):
                er = entry_ranks[rng.randrange(len(entry_ranks))]
                new_scores.append(-float(er) + rng.gauss(0.0, 2.0))
                new_active.append(True)
            # Re-entries of absent known games.
            for i in range(len(all_uids)):
                if active[i]:
                    continue
                if rng.random() < reenter_rate:
                    er = entry_ranks[rng.randrange(len(entry_ranks))]
                    score[i] = -float(er) + rng.gauss(0.0, 2.0)
                    active[i] = True
    denom = float(total_slot_sims) if total_slot_sims else 1.0
    probs = {}
    for i, uid in enumerate(all_uids):
        for p in range(places):
            if occ[i][p]:
                probs[(uid, p + 1)] = occ[i][p] / denom
    newcomer_share = [c / denom for c in newcomer_occ]
    return probs, newcomer_share, n_chart, {"sigma": sigma, "tau": tau, "curve": curve,
                                            "hazards": hazards, "entry_rate": entry_rate,
                                            "reenter_rate": reenter_rate}


def build_forecast(task, reads, n_sims=N_SIMS, seed=SEED):
    places = int(task["places"])
    probs, newcomer_share, n_chart, diag = simulate(task, reads, n_sims, seed)
    win_from = parse_time(task["window"]["from"])
    win_until = parse_time(task["window"]["until"])
    slots = int(task["window"]["slots"])
    t0 = reads[-1]["t"]
    horizons = [win_from + h * (win_until - win_from) / slots - t0 for h in range(slots)]
    reserve = newcomer_reserve(reads, horizons)
    diag["reserve"] = reserve
    diag["sim_newcomer_share"] = newcomer_share
    probabilities = []
    for k in range(1, places + 1):
        # Rescale known-game mass: the sim understates newcomer success, so
        # shrink known mass to (1 - reserve) and let the reserve cover
        # never-seen games via the unnamed share.
        sim_new = newcomer_share[k - 1]
        scale = (1.0 - reserve) / max(1.0 - sim_new, 1e-9)
        cands = sorted(
            ((p * scale, uid) for (uid, pl), p in probs.items() if pl == k),
            key=lambda x: (-x[0], x[1]),
        )
        named = []
        total = 0.0
        for p, uid in cands:
            # Water-fill: name while p beats the unnamed per-game value and
            # the newcomer reserve stays intact.
            unnamed = 1.0 - total
            per_game = unnamed / max(n_chart - len(named), 1)
            if p <= per_game:
                break
            if total + p > 1.0 - reserve:
                break
            named.append((uid, p))
            total += p
        if len(named) > 100:
            named = named[:100]
            total = sum(p for _, p in named)
        for uid, p in named:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    # Safety: no place may sum above 1 after rounding.
    sums = {}
    for row in probabilities:
        sums[row["place"]] = sums.get(row["place"], 0.0) + row["p"]
    for row in probabilities:
        if sums[row["place"]] > 1.0:
            over = sums[row["place"]] - 1.0
            row["p"] = round(max(row["p"] - over, 0.0), 6)
            sums[row["place"]] -= over
    # top10: order by place-weighted mass (top places weigh more).
    mass = {}
    for (uid, pl), p in probs.items():
        mass[uid] = mass.get(uid, 0.0) + p * (places + 1 - pl)
    top10 = [uid for uid, _ in sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]
    # Fall back to last chart order if the sim named too few games.
    if len(top10) < places:
        seen = set(top10)
        for uid in sorted(reads[-1]["chart"], key=lambda u: reads[-1]["chart"][u]):
            if uid not in seen:
                top10.append(uid)
                seen.add(uid)
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10}, diag


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_reads(input_dir)
    forecast, _ = build_forecast(task, reads)
    with open(os.path.join(output_dir, task["output"]), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
