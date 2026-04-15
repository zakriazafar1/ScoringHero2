import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from microarousal_pipeline_full import compute_morlet_tf_streaming, FREQS, BANDS

# ── instellingen ──
SUBJ    = "bnbd_nsr_01272"
NIGHT   = "T0_N1"
BASE_DIR   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output4")

SFREQ   = 256
PAD_SEC = 5

# ── paden ──
edf_path = BASE_DIR / SUBJ / f"{SUBJ}_{NIGHT}" / "sleepArchitecture" / f"{SUBJ}_{NIGHT}_psg.edf"
csv_path = OUTPUT_DIR / f"{SUBJ}_{NIGHT}_events.csv"

# ── laad EEG ──
print("EDF laden...")
raw   = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
eeg_L = raw.get_data(picks="EEG L psg-lp")[0]
eeg_R = raw.get_data(picks="EEG R psg-lp")[0]
eeg   = (eeg_L + eeg_R) / 2.0
del raw   # geheugen vrijmaken

# ── bereken ridge EENMALIG voor hele nacht ──
# dit is de sleutel: niet per event opnieuw berekenen
print("CWT berekenen voor hele nacht (één keer)...")
ridge_freq, _, _ = compute_morlet_tf_streaming(eeg, SFREQ, FREQS, BANDS)

# ── laad events ──
events = pd.read_csv(csv_path)
print(f"{len(events)} events gevonden")

# ── maak één grote PDF met alle events ──
# PDF is ideaal: één bestand, alle pagina's, kleine bestandsgrootte
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = OUTPUT_DIR / f"{SUBJ}_{NIGHT}_alle_events.pdf"

print(f"Plotten naar PDF...")
with PdfPages(pdf_path) as pdf:
    for i, row in events.iterrows():
        start_s = int(row['start_sec'] * SFREQ)
        end_s   = int(row['end_sec']   * SFREQ)
        pad_s   = int(PAD_SEC * SFREQ)

        win_start = max(0, start_s - pad_s)
        win_end   = min(len(eeg), end_s + pad_s)

        times        = np.arange(win_end - win_start) / SFREQ + (win_start / SFREQ)
        ev_start_rel = (start_s - win_start) / SFREQ + (win_start / SFREQ)
        ev_end_rel   = (end_s   - win_start) / SFREQ + (win_start / SFREQ)

        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

        # EEG
        axes[0].plot(times, eeg[win_start:win_end] * 1e6, lw=0.5, color="steelblue")
        axes[0].axvspan(ev_start_rel, ev_end_rel, alpha=0.25, color="orange")
        axes[0].axvline(ev_start_rel, color="orange", lw=1.2, linestyle="--")
        axes[0].axvline(ev_end_rel,   color="orange", lw=1.2, linestyle="--")
        axes[0].set_ylabel("EEG (µV)")
        axes[0].set_title(
            f"Event {i+1}/{len(events)}  |  {row['start_sec']:.1f}–{row['end_sec']:.1f}s  "
            f"|  duur {row['duration']:.1f}s  |  "
            f"ridge {row['mean_ridge_freq']:.1f} Hz  |  {row['artifact_label']}",
            fontsize=9
        )

        # ridge — slice uit al berekende array
        axes[1].plot(times, ridge_freq[win_start:win_end], lw=0.8, color="darkorange")
        axes[1].axvspan(ev_start_rel, ev_end_rel, alpha=0.25, color="orange")
        axes[1].axhline(8,  color="green", lw=0.8, linestyle=":", label="alpha (8 Hz)")
        axes[1].axhline(13, color="red",   lw=0.8, linestyle=":", label="beta (13 Hz)")
        axes[1].set_ylabel("Ridge freq (Hz)")
        axes[1].set_xlabel("Tijd (seconden)")
        axes[1].set_ylim(0, 35)
        axes[1].legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)   # geheugen vrijmaken na elke pagina

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(events)} events geplot...")

print(f"\nKlaar — PDF opgeslagen: {pdf_path}")