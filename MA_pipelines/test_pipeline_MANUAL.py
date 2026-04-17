import mne
import numpy as np
from pathlib import Path

# ── test 1: kan Python de libraries vinden? ──
print("Bibliotheken laden...")
try:
    import mne
    import numpy as np
    import pandas as pd
    print("  OK: mne, numpy, pandas gevonden")
except ImportError as e:
    print(f"  FOUT: {e}")
    print("  Oplossing: pip install mne numpy pandas")

# ── test 2: kan je een EDF bestand vinden? ──
print("\nEDF bestanden zoeken...")
base_dir = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
all_edf  = sorted(base_dir.rglob("*.edf"))
print(f"  Gevonden: {len(all_edf)} EDF bestanden")
if all_edf:
    print(f"  Eerste bestand: {all_edf[0]}")

# ── test 3: kan je het eerste EDF openen? ──
if all_edf:
    print("\nEerste EDF openen...")
    try:
        raw = mne.io.read_raw_edf(all_edf[0], preload=False, verbose=False)
        print(f"  OK: {raw.info['sfreq']} Hz, kanalen: {raw.ch_names}")
    except Exception as e:
        print(f"  FOUT: {e}")

# ── test 4: kleine CWT op nepdata ──
print("\nCWT testen op nepdata (5 seconden)...")
try:
    from microarousal_pipeline_full import compute_morlet_tf
    fake_eeg = np.random.randn(256 * 5)   # 5 seconden ruis
    freqs    = np.arange(0.5, 35.5, 0.5)
    power    = compute_morlet_tf(fake_eeg, 256, freqs)
    print(f"  OK: power matrix shape = {power.shape}")
    print(f"      verwacht: (70, 1280)")
except Exception as e:
    print(f"  FOUT: {e}")

print("\nTest klaar.")