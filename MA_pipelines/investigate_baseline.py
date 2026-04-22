"""
investigate_baseline.py
========================
Onderzoekt of het ridge_freq verschil tussen groepen een baseline-effect is
(hele nacht) of specifiek tijdens events optreedt.

Analyses:
1. Baseline ridge_freq (buiten events) vs event ridge_freq per persoon
2. Histogrammen van ridge_freq over hele nacht per groep
3. Binnen-persoon vergelijking: baseline vs event ridge_freq
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from microarousal_pipeline_MANUAL import detect_events

# ── Configuratie ──────────────────────────────────────────────────────────────

CACHE_DIR  = Path(r"C:\Users\zafar\Documents\bnbd_output_mc_params")
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output_mc_params_no_outliers")

ROLLING_SEC = 30
THRESHOLD   = 0.771
SFREQ       = 256.0
DS_FREQ     = 4.0
MIN_DUR     = 1.0
MAX_DUR     = 30.0
MERGE_GAP   = 1.0

GROUP_A_IDS = ["15103","16775","03554","05830","17688",
               "16956","16924","17322","19330"]   # lage angst, zonder outliers
GROUP_B_IDS = ["18500","20362","20736","21962","23343"]  # hoge angst, zonder outliers

COLOR_A = "#2196F3"
COLOR_B = "#F44336"

# ── Hulpfuncties ──────────────────────────────────────────────────────────────

def compute_freq_shift(ridge_freq, rolling_sec, sfreq=DS_FREQ):
    window   = int(rolling_sec * sfreq)
    series   = pd.Series(ridge_freq.astype(np.float64))
    baseline = series.rolling(window=window, min_periods=1).median().values
    return (ridge_freq - baseline).astype(np.float32)


def load_nights(subject_id):
    files = sorted(CACHE_DIR.glob(f"{subject_id}_T0_N*_ridge.npz"))
    nights = []
    for f in files:
        d  = np.load(f)
        ds = int(SFREQ / DS_FREQ)
        nights.append({
            "ridge_freq":  d["ridge_freq"][::ds].astype(np.float32),
            "ridge_power": d["ridge_power"][::ds].astype(np.float32),
        })
    return nights


def subject_split(subject_id):
    """
    Geeft per persoon (over alle nachten samengevoegd):
    - ridge_freq tijdens events
    - ridge_freq buiten events
    - freq_shift tijdens events
    """
    nights = load_nights(subject_id)
    if not nights:
        return None

    event_rf, baseline_rf, event_fs = [], [], []

    for night in nights:
        rf = night["ridge_freq"]
        rp = night["ridge_power"]
        fs = compute_freq_shift(rf, ROLLING_SEC)

        evts = detect_events(rf, rp, fs, DS_FREQ,
                             threshold=THRESHOLD,
                             min_dur=MIN_DUR, max_dur=MAX_DUR,
                             merge_gap_sec=MERGE_GAP)

        # masker: welke samples zijn event-samples?
        mask = np.zeros(len(rf), dtype=bool)
        for s, e in evts:
            mask[s:e] = True

        event_rf.extend(rf[mask].tolist())
        baseline_rf.extend(rf[~mask].tolist())
        event_fs.extend(fs[mask].tolist())

    return {
        "event_rf":    np.array(event_rf),
        "baseline_rf": np.array(baseline_rf),
        "event_fs":    np.array(event_fs),
    }


# ── Per persoon statistieken ──────────────────────────────────────────────────

def compute_all_stats(ids, group_label):
    rows = []
    for sid in tqdm(ids, desc=group_label):
        result = subject_split(sid)
        if result is None:
            continue
        rows.append({
            "subject":          sid,
            "group":            group_label,
            "med_baseline_rf":  float(np.median(result["baseline_rf"])) if len(result["baseline_rf"]) else np.nan,
            "med_event_rf":     float(np.median(result["event_rf"]))    if len(result["event_rf"])    else np.nan,
            "med_event_fs":     float(np.median(result["event_fs"]))    if len(result["event_fs"])    else np.nan,
            "n_event_samples":  len(result["event_rf"]),
            "n_baseline_samples": len(result["baseline_rf"]),
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_baseline_vs_event(df_a, df_b):
    """
    Per persoon: baseline ridge_freq vs event ridge_freq.
    Laat zien of het verschil al in de baseline zit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Baseline vs event ridge_freq per persoon", fontsize=12)

    metrics = [
        ("med_baseline_rf", "Mediane baseline ridge_freq (Hz)\n(buiten events)"),
        ("med_event_rf",    "Mediane event ridge_freq (Hz)\n(tijdens events)"),
        ("med_event_fs",    "Mediane freq_shift tijdens events (Hz)\n(event - eigen baseline)"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        a = df_a[col].dropna().values
        b = df_b[col].dropna().values

        rng = np.random.default_rng(0)
        for vals, pos, col_c in zip([a, b], [1, 2], [COLOR_A, COLOR_B]):
            jitter = rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(pos + jitter, vals, s=60, color=col_c, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.plot([pos - 0.2, pos + 0.2],
                    [np.median(vals), np.median(vals)],
                    color=col_c, lw=2.5, zorder=4)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Lage angst\n(n={len(a)})", f"Hoge angst\n(n={len(b)})"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

        # annoteer mediaan verschil
        if len(a) and len(b):
            diff = np.median(b) - np.median(a)
            ax.set_title(f"Mediaan verschil: {diff:+.2f} Hz", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "investigate_baseline_vs_event.png"
    plt.savefig(out, dpi=150)
    print(f"Plot opgeslagen: {out}")
    plt.show()


def plot_ridge_histograms(ids_a, ids_b):
    """
    Histogram van ridge_freq over hele nacht — één lijn per persoon,
    groepen in verschillende kleuren.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Verdeling ridge_freq over hele nacht (alle samples)", fontsize=12)

    bins = np.arange(0.5, 35.5, 0.5)

    for ax, (ids, color, label) in zip(axes, [
        (ids_a, COLOR_A, "Lage angst (NSR)"),
        (ids_b, COLOR_B, "Hoge angst (SAV)"),
    ]):
        group_hists = []
        for sid in ids:
            nights = load_nights(sid)
            if not nights:
                continue
            all_rf = np.concatenate([n["ridge_freq"] for n in nights])
            counts, _ = np.histogram(all_rf, bins=bins, density=True)
            group_hists.append(counts)
            ax.plot(bins[:-1], counts, color=color, alpha=0.3, lw=1)

        if group_hists:
            mean_hist = np.mean(group_hists, axis=0)
            ax.plot(bins[:-1], mean_hist, color=color, lw=2.5,
                    label=f"Groepsgemiddelde (n={len(group_hists)})")

        ax.set_xlabel("Ridge freq (Hz)")
        ax.set_ylabel("Dichtheid")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 20)

    plt.tight_layout()
    out = OUTPUT_DIR / "investigate_ridge_histograms.png"
    plt.savefig(out, dpi=150)
    print(f"Plot opgeslagen: {out}")
    plt.show()


# ── Hoofdscript ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Statistieken berekenen...")
    df_a = compute_all_stats(GROUP_A_IDS, "low_anxiety")
    df_b = compute_all_stats(GROUP_B_IDS, "high_anxiety")
    df   = pd.concat([df_a, df_b], ignore_index=True)

    print("\n" + "="*60)
    print("BASELINE vs EVENT ridge_freq per persoon")
    print("="*60)
    print(df[["subject","group","med_baseline_rf","med_event_rf","med_event_fs"]].to_string(index=False))

    print("\n" + "="*60)
    print("GROEPSGEMIDDELDEN (mediaan over personen)")
    print("="*60)
    for col, label in [
        ("med_baseline_rf", "Baseline ridge_freq"),
        ("med_event_rf",    "Event ridge_freq"),
        ("med_event_fs",    "Freq_shift tijdens events"),
    ]:
        med_a = df_a[col].median()
        med_b = df_b[col].median()
        print(f"{label}:")
        print(f"  Lage angst:  {med_a:.2f} Hz")
        print(f"  Hoge angst:  {med_b:.2f} Hz")
        print(f"  Verschil:    {med_b - med_a:+.2f} Hz")

    print("\nPlots maken...")
    plot_baseline_vs_event(df_a, df_b)
    plot_ridge_histograms(GROUP_A_IDS, GROUP_B_IDS)
