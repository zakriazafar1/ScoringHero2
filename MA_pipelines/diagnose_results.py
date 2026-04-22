"""
diagnose_results.py
====================
Diagnostiek op de gevonden optimale parameters.

1. Laad cache uit bnbd_output_mc_params
2. Draai detect_events met optimale params per persoon (over alle nachten)
3. Bereken per persoon: n_events, events/uur, mediane duur, mediane ridge_freq
4. Boxplots + individuele punten (swarm)
5. Bootstrap CI voor Cohen's d
6. Mediaan-gebaseerde effectmaat (Hodges-Lehmann)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from microarousal_pipeline_MANUAL import detect_events

# ── Configuratie ──────────────────────────────────────────────────────────────

CACHE_DIR  = Path(r"C:\Users\zafar\Documents\bnbd_output_mc_params_no_outliers")
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output_mc_params_no_outliers")

ROLLING_SEC = 60      # optimale waarde uit find_optimal_params
THRESHOLD   = 0.731      # optimale waarde uit find_optimal_params
SFREQ       = 256.0
DS_FREQ     = 4.0        # zelfde downsampling als find_optimal_params
MIN_DUR     = 1.0
MAX_DUR     = 30.0
MERGE_GAP   = 1.0

GROUP_A_IDS = [
    "15103",
    # "14359",   # outlier: 101 events/uur
    "16775",
    "03554",
    "05830",
    "17688",
    "16956",
    "16924",
    "17322",
    "18996",
    # "05578",   # outlier: mean_ridge_freq 11.09 Hz
    "19330"
]

GROUP_B_IDS = [
    "18500",
    "20362",
    "20736",
    # "21614",   # outlier: 18 events/uur, slechts 1 nacht
    "21962",
    "23343"
]

N_BOOTSTRAP = 2000

# ── Hulpfuncties ──────────────────────────────────────────────────────────────

def compute_freq_shift(ridge_freq, rolling_sec, sfreq=DS_FREQ):
    window   = int(rolling_sec * sfreq)
    series   = pd.Series(ridge_freq.astype(np.float64))
    baseline = series.rolling(window=window, min_periods=1).median().values
    return (ridge_freq - baseline).astype(np.float32)


def load_nights(subject_id):
    """Laad alle gecachede nachten voor één persoon."""
    files = sorted(CACHE_DIR.glob(f"{subject_id}_T0_N*_ridge.npz"))
    nights = []
    for f in files:
        d = np.load(f)
        ds = int(SFREQ / DS_FREQ)
        nights.append({
            "night":       f.stem,
            "ridge_freq":  d["ridge_freq"][::ds].astype(np.float32),
            "ridge_power": d["ridge_power"][::ds].astype(np.float32),
        })
    return nights


def events_for_night(night, rolling_sec, threshold):
    rf  = night["ridge_freq"]
    rp  = night["ridge_power"]
    fs  = compute_freq_shift(rf, rolling_sec)
    evs = detect_events(rf, rp, fs, DS_FREQ,
                        threshold=threshold,
                        min_dur=MIN_DUR, max_dur=MAX_DUR,
                        merge_gap_sec=MERGE_GAP)
    records = []
    for s, e in evs:
        records.append({
            "duration":   (e - s) / DS_FREQ,
            "ridge_freq": float(rf[s:e].mean()),
            "freq_shift": float(fs[s:e].mean()),
        })
    hours = len(rf) / DS_FREQ / 3600
    return records, hours


# ── Per persoon aggregeren ────────────────────────────────────────────────────

def compute_subject_stats(subject_ids, group_label):
    rows = []
    for sid in tqdm(subject_ids, desc=f"  {group_label}"):
        nights = load_nights(sid)
        if not nights:
            print(f"    Geen cache: {sid}")
            continue

        all_events = []
        total_hours = 0.0
        for night in nights:
            evs, hrs = events_for_night(night, ROLLING_SEC, THRESHOLD)
            all_events.extend(evs)
            total_hours += hrs

        n_events = len(all_events)
        eph      = n_events / total_hours if total_hours > 0 else 0

        if all_events:
            df_ev = pd.DataFrame(all_events)
            med_dur   = df_ev["duration"].median()
            med_freq  = df_ev["ridge_freq"].median()
            med_shift = df_ev["freq_shift"].median()
            mean_dur  = df_ev["duration"].mean()
            mean_freq = df_ev["ridge_freq"].mean()
        else:
            med_dur = med_freq = med_shift = mean_dur = mean_freq = np.nan

        rows.append({
            "subject":      sid,
            "group":        group_label,
            "n_events":     n_events,
            "total_hours":  round(total_hours, 2),
            "events_per_hr": round(eph, 1),
            "med_duration": round(med_dur,  2) if not np.isnan(med_dur)  else np.nan,
            "med_ridge_freq": round(med_freq, 2) if not np.isnan(med_freq) else np.nan,
            "med_freq_shift": round(med_shift, 3) if not np.isnan(med_shift) else np.nan,
            "mean_duration": round(mean_dur, 2) if not np.isnan(mean_dur) else np.nan,
            "mean_ridge_freq": round(mean_freq, 2) if not np.isnan(mean_freq) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Bootstrap Cohen's d ───────────────────────────────────────────────────────

def bootstrap_cohens_d(a, b, n=N_BOOTSTRAP, seed=42):
    """Bootstrap CI voor Cohen's d (b - a)."""
    rng = np.random.default_rng(seed)
    ds  = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        pooled = np.sqrt(((len(sa)-1)*sa.std(ddof=1)**2 +
                          (len(sb)-1)*sb.std(ddof=1)**2) /
                         (len(sa)+len(sb)-2))
        ds.append((sb.mean() - sa.mean()) / (pooled + 1e-10))
    return np.array(ds)


def hodges_lehmann(a, b):
    """Mediaan van alle paarsgewijze (b-a) verschillen — robuuste effectmaat."""
    diffs = np.array([bv - av for bv in b for av in a])
    return float(np.median(diffs))


# ── Plots ─────────────────────────────────────────────────────────────────────

COLOR_A = "#2196F3"   # blauw — lage angst
COLOR_B = "#F44336"   # rood  — hoge angst

def swarm_box(ax, data_a, data_b, ylabel, title):
    """Boxplot + individuele punten voor twee groepen."""
    positions = [1, 2]
    bp = ax.boxplot([data_a, data_b], positions=positions,
                    widths=0.4, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=""))

    for patch, color in zip(bp["boxes"], [COLOR_A, COLOR_B]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # individuele punten (jitter)
    rng = np.random.default_rng(0)
    for vals, pos, col in zip([data_a, data_b], positions, [COLOR_A, COLOR_B]):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, color=col, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Lage angst\n(NSR, n=A)", "Hoge angst\n(SAV, n=B)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def plot_bootstrap_d(ax, boot_d, observed_d, title):
    """Histogram van bootstrap Cohen's d met CI."""
    ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])
    ax.hist(boot_d, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(observed_d, color="red",    lw=2, label=f"Geobserveerd d = {observed_d:.2f}")
    ax.axvline(ci_lo,      color="black",  lw=1.5, linestyle="--",
               label=f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axvline(ci_hi,      color="black",  lw=1.5, linestyle="--")
    ax.axvline(0,          color="gray",   lw=1, linestyle=":")
    ax.set_xlabel("Cohen's d (hoge − lage angst)")
    ax.set_ylabel("Bootstrap frequentie")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ── Hoofdscript ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Statistieken per persoon berekenen...")
    df_a = compute_subject_stats(GROUP_A_IDS, "low_anxiety")
    df_b = compute_subject_stats(GROUP_B_IDS, "high_anxiety")
    df   = pd.concat([df_a, df_b], ignore_index=True)

    csv_out = OUTPUT_DIR / "subject_stats.csv"
    df.to_csv(csv_out, index=False)

    print("\n" + "="*60)
    print("RUWE DATA PER PERSOON")
    print("="*60)
    print(df.to_string(index=False))

    # ── groepswaarden ophalen ──
    METRICS = [
        ("n_events",      "Totaal events"),
        ("events_per_hr", "Events / uur"),
        ("med_duration",  "Mediane duur (sec)"),
        ("med_ridge_freq","Mediane ridge freq (Hz)"),
        ("med_freq_shift","Mediane freq shift (Hz)"),
    ]

    a_vals = {m: df_a[m].dropna().values for m, _ in METRICS}
    b_vals = {m: df_b[m].dropna().values for m, _ in METRICS}

    # ── Cohen's d + bootstrap CI + Hodges-Lehmann ──
    print("\n" + "="*60)
    print("EFFECT SIZES (mediaan-gebaseerd, hoge − lage angst)")
    print("="*60)
    for metric, label in METRICS:
        a = a_vals[metric]
        b = b_vals[metric]
        if len(a) < 2 or len(b) < 2:
            continue

        hl   = hodges_lehmann(a, b)
        boot = bootstrap_cohens_d(a, b)
        obs  = (b.mean() - a.mean()) / (np.sqrt(((len(a)-1)*a.std(ddof=1)**2 +
               (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2)) + 1e-10)
        ci   = np.percentile(boot, [2.5, 97.5])
        _, pval = mannwhitneyu(a, b, alternative="two-sided")

        print(f"\n{label}:")
        print(f"  Mediaan lage angst:  {np.median(a):.2f}")
        print(f"  Mediaan hoge angst:  {np.median(b):.2f}")
        print(f"  Hodges-Lehmann:      {hl:.3f}")
        print(f"  Cohen's d:           {obs:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"  Mann-Whitney p:      {pval:.3f}")

    # ── Figuur 1: swarm + boxplots ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Diagnostiek — threshold={THRESHOLD} Hz, rolling={ROLLING_SEC}s",
                 fontsize=12, fontweight="bold")

    plot_configs = [
        ("n_events",       "Totaal events",          axes[0, 0]),
        ("events_per_hr",  "Events / uur",            axes[0, 1]),
        ("med_duration",   "Mediane duur (sec)",      axes[0, 2]),
        ("med_ridge_freq", "Mediane ridge freq (Hz)", axes[1, 0]),
        ("med_freq_shift", "Mediane freq shift (Hz)", axes[1, 1]),
    ]

    for metric, label, ax in plot_configs:
        a = a_vals[metric]
        b = b_vals[metric]
        n_a, n_b = len(a), len(b)
        ax.get_xticklabels  # placeholder
        swarm_box(ax, a, b, label, label)
        # fix label met juiste n
        ax.set_xticklabels([f"Lage angst\n(n={n_a})", f"Hoge angst\n(n={n_b})"])

    axes[1, 2].axis("off")  # lege cel

    patch_a = mpatches.Patch(color=COLOR_A, alpha=0.5, label="Lage angst (NSR)")
    patch_b = mpatches.Patch(color=COLOR_B, alpha=0.5, label="Hoge angst (SAV)")
    fig.legend(handles=[patch_a, patch_b], loc="lower right", fontsize=10)

    plt.tight_layout()
    out1 = OUTPUT_DIR / "diagnose_swarm.png"
    plt.savefig(out1, dpi=150)
    print(f"\nPlot opgeslagen: {out1}")
    plt.show()

    # ── Figuur 2: bootstrap Cohen's d per maat ──
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle("Bootstrap Cohen's d (hoge − lage angst)", fontsize=12)

    for idx, (metric, label) in enumerate(METRICS):
        ax  = axes2[idx // 3][idx % 3]
        a   = a_vals[metric]
        b   = b_vals[metric]
        if len(a) < 2 or len(b) < 2:
            ax.axis("off")
            continue
        boot = bootstrap_cohens_d(a, b)
        obs  = (b.mean() - a.mean()) / (np.sqrt(((len(a)-1)*a.std(ddof=1)**2 +
               (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2)) + 1e-10)
        plot_bootstrap_d(ax, boot, obs, label)

    axes2[1][2].axis("off")
    plt.tight_layout()
    out2 = OUTPUT_DIR / "diagnose_bootstrap_d.png"
    plt.savefig(out2, dpi=150)
    print(f"Plot opgeslagen: {out2}")
    plt.show()
