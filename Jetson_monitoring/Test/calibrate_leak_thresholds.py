#%% ================================================================
#  Leak-Alert Threshold Calibration
# ====================================================================
"""
Turns an overnight leak-test CSV (test_leak_overnight.py's output, in
leak_test_logs/) into calibrated values for the three warning-point
thresholds leak_alert.py's Tier-2 (indirect, BME280-derived) check
compares against: LEAK_PRESSURE_DEVIATION_THRESHOLD_MBAR,
LEAK_DEW_POINT_RISE_THRESHOLD_C, and LEAK_TEMP_DEVIATION_THRESHOLD_C
(config.py) -- previously all PLACEHOLDER guesses with no real baseline
behind them.

WHY THIS ISN'T A STRAIGHT "MAX OBSERVED VALUE" CALIBRATION
------------------------------------------------------------------
An overnight leak test starts from a hand-pumped-down enclosure, so the
CSV is dominated by two large, real, and entirely benign trends that
have nothing to do with sensor noise or an actual leak: the internal
pressure relaxing back toward ambient over several hours, and the
electronics/enclosure slowly warming up (which also drags relative
humidity and dew point along with it). leak_alert.check_for_leak()
never sees that in real deployment -- the enclosure is sealed once, at
equilibrium, and stays there. So calibrating straight off the raw
15-minute rise/deviation numbers in this CSV would set thresholds high
enough to only catch a leak as dramatic as a fresh pump-down, which
defeats the purpose.

Instead, each signal is first detrended with a long (DETREND_WINDOW_S)
centered rolling median -- long enough to absorb the multi-hour
decay/warm-up curves, short enough that a genuine ~15-minute leak
signature would still stand out in what's left over. What's left after
subtracting that trend is treated as the real noise floor
check_for_leak has to tolerate, and *that* residual is what gets run
through the exact same windowed-rise comparison check_for_leak()
actually performs (current value minus the oldest value still inside
the trailing LEAK_WARNING_WINDOW_S), before taking a high percentile
of it plus a safety margin as the suggested threshold.

USAGE
------------------------------------------------------------------
    python3 calibrate_leak_thresholds.py [csv_path] [--percentile P] [--margin M] [--apply]

With no csv_path, uses the newest leak_test_*.csv in leak_test_logs/.
--apply patches the suggested numbers directly into Monitor/config.py
(each LEAK_*_THRESHOLD_* line must already exist -- add a placeholder
line by hand the first time a new threshold is introduced).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MONITOR_DIR = Path(__file__).resolve().parent.parent / "Monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))
import config
import leak_alert

LOG_DIR = Path(__file__).resolve().parent / "leak_test_logs"

# Sanity ranges for dropping garbled serial lines before they can corrupt the fit --
# same idea (and same temp range) as test_leak_overnight.py's INTERNAL_TEMP_SANE_RANGE_C.
HUMIDITY_SANE_RANGE_PCT = (0.0, 100.0)
TEMP_SANE_RANGE_C = (-10.0, 60.0)
PRESSURE_SANE_RANGE_PA = (30_000.0, 110_000.0)  # generous enough to cover a hard vacuum pump-down through 1 atm

DETREND_WINDOW_S = 4 * config.LEAK_WARNING_WINDOW_S  # 1 hour by default -- see module docstring
NOISE_PERCENTILE = 99.5  # how far into the residual noise distribution's tail to anchor the
# threshold -- high enough to ignore the odd sensor glitch that slipped past the sanity filter,
# not so high (100th = the single worst sample) that one outlier sets the whole threshold.
SAFETY_MARGIN = 2.0  # final threshold = observed noise ceiling * this -- real margin against
# "cries wolf" false positives, without hand-guessing a number (see config.py's old PLACEHOLDER).

# (dataframe column, human label, config.py constant name)
SIGNALS = [
    ("pressure_mbar", "Pressure", "LEAK_PRESSURE_DEVIATION_THRESHOLD_MBAR"),
    ("dew_point_c", "Dew point", "LEAK_DEW_POINT_RISE_THRESHOLD_C"),
    ("temp_int_c", "Temperature", "LEAK_TEMP_DEVIATION_THRESHOLD_C"),
]


def load_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.dropna(subset=["t_rel_s", "p_int_pa", "temp_int_c", "humidity_pct"])
    sane = (
        df["humidity_pct"].between(*HUMIDITY_SANE_RANGE_PCT)
        & df["temp_int_c"].between(*TEMP_SANE_RANGE_C)
        & df["p_int_pa"].between(*PRESSURE_SANE_RANGE_PA)
    )
    rejected = before - int(sane.sum())
    df = df[sane].sort_values("t_rel_s").reset_index(drop=True)
    print(f"Loaded {before} rows, rejected {rejected} outside sane sensor ranges (serial glitches), "
          f"{len(df)} remain.")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pressure_mbar"] = df["p_int_pa"] / 100.0
    df["dew_point_c"] = [
        leak_alert.dew_point_c(t, h) for t, h in zip(df["temp_int_c"], df["humidity_pct"])
    ]
    return df


def detrend(df: pd.DataFrame, column: str) -> np.ndarray:
    """Residual = raw - a long centered rolling median, so the real multi-hour pump-down
    decay / warm-up trend this test protocol itself creates gets treated as trend, not noise
    (see module docstring)."""
    s = pd.Series(df[column].to_numpy(), index=pd.to_timedelta(df["t_rel_s"], unit="s"))
    trend = s.rolling(f"{DETREND_WINDOW_S}s", center=True, min_periods=1).median()
    return (s - trend).to_numpy()


def windowed_rise(t_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Reproduces leak_alert.check_for_leak's exact comparison: current value minus the
    oldest value still inside the trailing `window_s` history -- a single lagged difference,
    not a smoothed slope, since that's what actually gets thresholded in production. Only
    populated where at least half the window has elapsed, matching check_for_leak's own
    "not enough history yet" guard."""
    j = np.searchsorted(t_s, t_s - window_s, side="left")
    span = t_s - t_s[j]
    valid = span >= 0.5 * window_s
    rise = np.full(len(t_s), np.nan)
    rise[valid] = values[valid] - values[j[valid]]
    return rise


def suggest_threshold(rise: np.ndarray, percentile: float, margin: float) -> float:
    rise = rise[~np.isnan(rise)]
    ceiling = np.percentile(np.abs(rise), percentile)
    return ceiling * margin


def apply_to_config(results: dict, config_path: Path) -> None:
    text = config_path.read_text()
    for name, value in results.items():
        pattern = rf"^{name} = .+$"
        new_text, n = re.subn(pattern, f"{name} = {value:.2f}", text, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(f"{name} not found in {config_path} -- add a placeholder line "
                                f"for it first, then re-run with --apply.")
        text = new_text
    config_path.write_text(text)

#%%
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv_path", nargs="?", default=None, help="Overnight leak test CSV (test_leak_overnight.py's output) Defaults to the newest file in leak_test_logs/.")
parser.add_argument("--percentile", type=float, default=NOISE_PERCENTILE)
parser.add_argument("--margin", type=float, default=SAFETY_MARGIN)
parser.add_argument("--apply", action="store_true", help="Patch the suggested values directly into Monitor/config.py.")
#%%
# parse_known_args, not parse_args: running this via VSCode's Jupyter interactive window
# (the #%% cells) launches an ipykernel that appends its own --f=<connection file> to
# sys.argv -- parse_args would reject that as an unrecognized argument and exit.
args, _ = parser.parse_known_args()
#%%
if args.csv_path:
    csv_path = Path(args.csv_path)
else:
    candidates = sorted(LOG_DIR.glob("leak_test_*.csv"))
    if not candidates:
        sys.exit(f"No leak_test_*.csv files found in {LOG_DIR}")
    csv_path = candidates[-1]

print(f"Calibrating from: {csv_path}")
df = load_samples(csv_path)
if len(df) < 100:
    sys.exit("Not enough clean samples to calibrate from -- was this a full overnight run?")
df = add_derived_columns(df)

duration_hr = (df["t_rel_s"].iloc[-1] - df["t_rel_s"].iloc[0]) / 3600.0
window_min = config.LEAK_WARNING_WINDOW_S / 60
print(f"Test duration: {duration_hr:.1f}h | alert window: {window_min:.0f} min | "
      f"detrend window: {DETREND_WINDOW_S / 60:.0f} min")
if duration_hr < 4:
    print("WARNING: less than 4h of data -- the detrend window and noise-floor percentile "
          "below will be poorly estimated. Prefer a longer overnight run if possible.")

t_s = df["t_rel_s"].to_numpy()
results = {}

for column, label, config_name in SIGNALS:
    raw_rise = windowed_rise(t_s, df[column].to_numpy(), config.LEAK_WARNING_WINDOW_S)
    residual = detrend(df, column)
    residual_rise = windowed_rise(t_s, residual, config.LEAK_WARNING_WINDOW_S)
    threshold = suggest_threshold(residual_rise, args.percentile, args.margin)
    results[config_name] = threshold

    raw_p = np.nanpercentile(np.abs(raw_rise), args.percentile)
    noise_p = threshold / args.margin
    print(f"\n{label} ({column}):")
    print(f"  raw {args.percentile:g}th pct |{window_min:.0f}-min change| (trend-contaminated): {raw_p:.3f}")
    print(f"  detrended noise-floor {args.percentile:g}th pct: {noise_p:.3f}")
    print(f"  -> suggested {config_name} = {threshold:.2f}  ({args.margin:g}x margin)")
print("\nSuggested config.py values:")
for name, value in results.items():
    print(f"  {name} = {value:.2f}")

if args.apply:
    config_path = config.MONITOR_DIR / "config.py"
    apply_to_config(results, config_path)
    print(f"\nPatched {config_path}")
else:
    print("\nRe-run with --apply to patch these directly into Monitor/config.py.")


