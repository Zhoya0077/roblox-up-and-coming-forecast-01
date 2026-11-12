# Roblox Up-and-Coming chart forecaster

Monte Carlo latent-score forecast of the Up-and-Coming top 10 over the
72-hour scored window. One forecast, scored at every hourly read, so the
output is the time-averaged probability of each game at each place.

## Model

- Each game ever seen carries a latent score, initialized from its rank at
  the last history read. Scores follow an Ornstein-Uhlenbeck process whose
  volatility and reversion time are fitted to the history's own rank-change
  curve (std of rank change vs lag), so the model re-calibrates itself on
  whatever history it is given. Hourly shocks carry a small jump mixture,
  because the chart moves in bursts: mostly frozen, occasionally reshuffled.
- A shrunk rank-velocity drift nudges scores along each game's recent trend
  and decays over about a day. Player counts are deliberately unused: in
  this history, playing-count changes do not predict rank changes at all,
  while rank momentum does (weakly).
- Games exit by a per-hour hazard fitted by rank band (bottom-ranked games
  fall off far more often), with a mild tilt for older games when a created
  date is known. There is no hard 28-day cliff: the data shows old games
  routinely staying on the chart.
- New games enter at the fitted hourly rate at ranks drawn from the observed
  entry-rank distribution; absent games can re-enter at a fitted rate.
- The chart is simulated hourly from the last read through the window end
  (seeded RNG, fully deterministic). Place occupancy is averaged over all
  slots and simulations.

## From simulations to probabilities

- Newcomers are the hardest part: the simulations understate how often a
  never-seen game reaches the top 10, so the unnamed reserve per place is
  set from the history's own never-seen share curve (share of top-10 slots
  held by games never seen before, vs horizon, averaged over the window).
  Known-game mass is rescaled to fit, and the reserve is left unnamed so
  unseen games share it evenly at each read.
- Games are named at a place only while their probability beats the unnamed
  per-game value (water-filling); the rest of the mass stays unnamed.
- `top10` orders games by place-weighted simulated mass.

## What it does not do

- No network, no hardcoded dates: window, places, and all calibration come
  from task.json and history.ndjson at run time.
- No playing-count, vote, or genre features: archive reads carry none, and
  the rank history alone carries the signal.
