import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from microarousal_pipeline_full import compute_morlet_tf_streaming, FREQS, BANDS

# ── instellingen ──
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output4")
BASE_DIR   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
PLOT_DIR   = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SFREQ   = 256
PAD_SEC = 5
N_PLOT  = None

# zoek alle event CSV bestanden (niet de master)
all_csv = sorted(f for f in OUTPUT_DIR.glob("*_events.csv")
                 if "master" not in f.name)

print(f"Gevonden: {len(all_csv)} event bestanden")

for csv_path in all_csv:
    # ── reconstrueer EDF pad uit CSV naam ──
    # csv naam: bnbd_nsr_01272_T0_N1_events.csv
    # edf pad:  .../bnbd_nsr_01272/bnbd_nsr_01272_T0_N1/sleepArchitecture/bnbd_nsr_01272_T0_N1_psg.edf
    stem     = csv_path.stem.replace("_events", "")   # bnbd_nsr_01272_T0_N1
    parts    = stem.split("_")                         # ['bnbd','nsr','01272','T0','N1']
    subj     = f"bnbd_nsr_{parts[2]}"                 # bnbd_nsr_01272
    night    = f"{parts[3]}_{parts[4]}"               # T0_N1
    edf_path = BASE_DIR / subj / f"{subj}_{night}" / "sleepArchitecture" / f"{subj}_{night}_psg.edf"

    if not edf_path.exists():
        print(f"  EDF niet gevonden: {edf_path.name}, overgeslagen")
        continue

    print(f"\nVerwerken: {stem}")

    # ── laad EEG ──
    try:
        raw   = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        eeg_L = raw.get_data(picks="EEG L psg-lp")[0]
        eeg_R = raw.get_data(picks="EEG R psg-lp")[0]
        eeg   = (eeg_L + eeg_R) / 2.0
    except Exception as e:
        print(f"  FOUT bij laden EDF: {e}")
        continue

    # ── laad events en pak random sample ──
    events = pd.read_csv(csv_path)
    sample = events.sample(min(N_PLOT, len(events))).reset_index(drop=True)
    print(f"  Plotten: {len(sample)} van {len(events)} events")

    # ── maak submap per nacht ──
    night_plot_dir = PLOT_DIR / stem
    night_plot_dir.mkdir(exist_ok=True)

    for i, row in sample.iterrows():
        start_s = int(row['start_sec'] * SFREQ)
        end_s   = int(row['end_sec']   * SFREQ)
        pad_s   = int(PAD_SEC * SFREQ)

        win_start = max(0, start_s - pad_s)
        win_end   = min(len(eeg), end_s + pad_s)
        eeg_win   = eeg[win_start:win_end]

        times        = np.arange(len(eeg_win)) / SFREQ + (win_start / SFREQ)
        ev_start_rel = (start_s - win_start) / SFREQ + (win_start / SFREQ)
        ev_end_rel   = (end_s   - win_start) / SFREQ + (win_start / SFREQ)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        # EEG
        axes[0].plot(times, eeg_win * 1e6, lw=0.6, color="steelblue")
        axes[0].axvspan(ev_start_rel, ev_end_rel, alpha=0.25, color="orange", label="event")
        axes[0].axvline(ev_start_rel, color="orange", lw=1.5, linestyle="--")
        axes[0].axvline(ev_end_rel,   color="orange", lw=1.5, linestyle="--")
        axes[0].set_ylabel("EEG (µV)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(
            f"{stem}  |  event {i+1}  |  "
            f"{row['start_sec']:.1f}–{row['end_sec']:.1f}s  |  "
            f"duur {row['duration']:.1f}s  |  "
            f"ridge {row['mean_ridge_freq']:.1f} Hz  |  "
            f"{row['artifact_label']}",
            fontsize=9
        )

        # ridge
        ridge_freq, _, _ = compute_morlet_tf_streaming(eeg_win, SFREQ, FREQS, BANDS)
        axes[1].plot(times, ridge_freq, lw=0.8, color="darkorange")
        axes[1].axvspan(ev_start_rel, ev_end_rel, alpha=0.25, color="orange")
        axes[1].axhline(8,  color="green", lw=0.8, linestyle=":", label="alpha (8 Hz)")
        axes[1].axhline(13, color="red",   lw=0.8, linestyle=":", label="beta (13 Hz)")
        axes[1].set_ylabel("Ridge freq (Hz)")
        axes[1].set_xlabel("Tijd (seconden)")
        axes[1].set_ylim(0, 35)
        axes[1].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        fname = night_plot_dir / f"event_{i+1:03d}_{row['artifact_label']}.png"
        plt.savefig(fname, dpi=120)
        plt.close()

    print(f"  Opgeslagen in: {night_plot_dir}")

print(f"\nKlaar — alle plots in: {PLOT_DIR}")