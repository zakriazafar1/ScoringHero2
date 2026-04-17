"""
find_optimal_params.py
=======================
Zoekt optimale ROLLING_SEC en AROUSAL_FREQ_THRESHOLD via 2D Monte Carlo.

Strategie:
- Gebruik groep A (N=12, lage angst) en groep B (N=6, hoge angst)
- Cache ridge_freq (volledige resolutie) eenmalig per proefpersoon
- Monte Carlo over (ROLLING_SEC, THRESHOLD):
    * ROLLING_SEC: getest op discrete grid, freq_shift vooraf berekend (snel)
    * THRESHOLD: uniform random gesampled
    * Bootstrap: per iteratie 30× subsample 6 uit 12 van groep A → stabiele schatting
- Objectief (features versie): gemiddelde Cohen's d over event-kenmerken × consistentie
- Geeft heatmap + JSON met optimale parameters

Gebruik:
    python find_optimal_params.py

Daarna: kopieer de waarden uit optimal_params_features.json naar microarousal_pipeline_MANUAL.py.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import json
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# Configuratie
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIRS = [
    Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR"),
    Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\SAV"),
]
CACHE_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output_mc_params")
CACHE_DIR.mkdir(exist_ok=True)

# ── Proefpersoon groepen ──────────────────────────────────────────────────────
GROUP_A_IDS = [
    "15103",
    "14359",
    "16775",
    "03554",
    "05830",
    "17688",
    "16956",
    "16924",
    "17322",
    "18996",
    "05578",
    "19330"
]

GROUP_B_IDS = [
    "18500",
    "20362",
    "20736",
    "21614",
    "21962",
    "23343"
]

# ── Signaalparameters ─────────────────────────────────────────────────────────
EEG_CH = ['EEG L psg-lp', 'EEG R psg-lp']
EMG_CH = ['EEG L psg-emg', 'EEG R psg-emg']
MOV_CH = ['dX', 'dY', 'dZ']
ALL_CH = EEG_CH + EMG_CH + MOV_CH
SFREQ  = 256.0
FREQS  = np.arange(0.5, 35.5, 0.5)
BANDS  = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 35.0),
}

# ── Monte Carlo zoekbereik ────────────────────────────────────────────────────
N_MONTE_CARLO = 500
ROLLING_GRID  = [20, 30, 45, 60, 90, 120, 150, 180]   # seconden
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 3.0

# ── Bootstrap configuratie ────────────────────────────────────────────────────
N_BOOTSTRAP = 30

# ── Downsampling voor MC ──────────────────────────────────────────────────────
DS_FREQ = 4.0   # Hz

# ── Event detectie grenzen ────────────────────────────────────────────────────
MIN_DUR_SEC   = 2.0
MAX_DUR_SEC   = 30.0
MERGE_GAP_SEC = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Hulpfuncties — laden en CWT
# ══════════════════════════════════════════════════════════════════════════════

def find_edfs(subject_number: str) -> list[Path]:
    """Zoekt alle T0 EDF bestanden voor een subject over beide BASE_DIRS."""
    found = []
    for base_dir in BASE_DIRS:
        for subject_folder in base_dir.glob(f"bnbd_*_{subject_number}"):
            for night_folder in subject_folder.glob("*_T0_N*"):
                sleep_arch = night_folder / "sleepArchitecture"
                if sleep_arch.is_dir():
                    found.extend(sleep_arch.glob("*.edf"))
    return sorted(found)


def load_and_filter(edf_path: Path) -> dict:
    """Laadt EDF en filtert de signalen."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    missing = [ch for ch in ALL_CH if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Kanalen ontbreken: {missing}")
    raw.pick(ALL_CH)
    raw.load_data(verbose=False)
    raw._data = raw._data.astype(np.float64)
    raw.filter(l_freq=0.5, h_freq=35.0, picks=EEG_CH, verbose=False)
    h_emg = min(100.0, SFREQ / 2 - 1)
    raw.filter(l_freq=10.0, h_freq=h_emg, picks=EMG_CH, verbose=False)
    raw.apply_function(lambda x: x - np.mean(x), picks=MOV_CH, verbose=False)
    return {ch: raw.get_data(picks=ch)[0] for ch in ALL_CH}


def compute_morlet_tf_streaming(signal):
    """
    Berekent CWT streaming (één frequentie per keer).
    Geeft: ridge_freq (n_samples,), ridge_power (n_samples,)
    """
    freqs        = FREQS
    n_cycles_arr = np.maximum(3.0, freqs / 2.0)
    n_samples    = len(signal)
    signal       = signal - np.mean(signal)
    signal_fft   = np.fft.fft(signal)
    fft_freqs    = np.fft.fftfreq(n_samples, d=1.0 / SFREQ)

    ridge_max = np.full(n_samples, -np.inf, dtype=np.float32)
    ridge_idx = np.zeros(n_samples, dtype=np.uint8)

    for i, freq in enumerate(tqdm(freqs, desc="CWT", leave=False)):
        sigma_f     = freq / n_cycles_arr[i]
        wavelet_fft = np.exp(-0.5 * ((fft_freqs - freq) / sigma_f) ** 2)
        analytic    = np.fft.ifft(signal_fft * wavelet_fft)
        power_row   = (np.abs(analytic) ** 2).astype(np.float32)
        better      = power_row > ridge_max
        ridge_max[better] = power_row[better]
        ridge_idx[better] = i

    return freqs[ridge_idx], ridge_max


# ══════════════════════════════════════════════════════════════════════════════
# Stap 1 — cache ridge_freq voor alle proefpersonen
# ══════════════════════════════════════════════════════════════════════════════

def night_key(edf_path: Path) -> str:
    """Extraheert een unieke sleutel uit het EDF pad: bijv. 'T0_N1'."""
    folder = edf_path.parent.parent.name
    for part in folder.split("_"):
        if part.startswith("N") and part[1:].isdigit():
            return f"T0_{part}"
    return folder


def cache_subject(subject_number: str, group_label: str) -> int:
    """Berekent en slaat ridge_freq op voor alle T0 nachten van één proefpersoon."""
    edf_paths = find_edfs(subject_number)
    if not edf_paths:
        print(f"  [{group_label}] {subject_number}: geen EDF bestanden gevonden")
        return 0

    cached = 0
    for edf_path in edf_paths:
        nk         = night_key(edf_path)
        cache_file = CACHE_DIR / f"{subject_number}_{nk}_ridge.npz"

        if cache_file.exists():
            print(f"  Cache al aanwezig: {cache_file.name}")
            cached += 1
            continue

        print(f"  [{group_label}] {subject_number} {nk} — {edf_path.name}")
        try:
            signals    = load_and_filter(edf_path)
            eeg_avg    = (signals[EEG_CH[0]] + signals[EEG_CH[1]]) / 2.0
            ridge_freq, ridge_power = compute_morlet_tf_streaming(eeg_avg)

            emg_avg = (np.abs(signals[EMG_CH[0]]) + np.abs(signals[EMG_CH[1]])) / 2.0
            acc     = np.sqrt(signals[MOV_CH[0]]**2 +
                              signals[MOV_CH[1]]**2 +
                              signals[MOV_CH[2]]**2)

            np.savez_compressed(
                cache_file,
                ridge_freq  = ridge_freq.astype(np.float32),
                ridge_power = ridge_power.astype(np.float32),
                emg         = emg_avg.astype(np.float32),
                acc         = acc.astype(np.float32),
            )
            print(f"    Opgeslagen: {cache_file.name}")
            cached += 1
        except Exception as e:
            print(f"    FOUT: {e}")

    return cached


def run_step1_cache():
    """Cache ridge data voor alle nachten van alle proefpersonen."""
    print("\n" + "="*60)
    print("STAP 1 — Ridge data cachen (alle T0 nachten)")
    print("="*60)

    if not GROUP_A_IDS and not GROUP_B_IDS:
        raise ValueError("Vul GROUP_A_IDS en GROUP_B_IDS in bovenaan het script.")

    total_nights = 0
    for sid in GROUP_A_IDS:
        total_nights += cache_subject(sid, "A (lage angst)")
    for sid in GROUP_B_IDS:
        total_nights += cache_subject(sid, "B (hoge angst)")

    print(f"\nStap 1 klaar — {total_nights} nachten gecached")


# ══════════════════════════════════════════════════════════════════════════════
# Gedeelde hulpfuncties voor Stap 2 (beide MC versies)
# ══════════════════════════════════════════════════════════════════════════════

def compute_freq_shift(ridge_freq: np.ndarray, rolling_sec: float,
                       sfreq: float = SFREQ) -> np.ndarray:
    """Berekent freq_shift t.o.v. rolling mediaan baseline."""
    window   = int(rolling_sec * sfreq)
    series   = pd.Series(ridge_freq)
    baseline = series.rolling(window=window, min_periods=1).median().values
    return (ridge_freq - baseline).astype(np.float32)


def load_cached_nights(subject_number: str) -> list[np.ndarray]:
    """Laadt gecachede ridge_freq voor alle beschikbare nachten van een subject."""
    files = sorted(CACHE_DIR.glob(f"{subject_number}_T0_N*_ridge.npz"))
    return [np.load(f)['ridge_freq'] for f in files]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: gestandaardiseerd gemiddeld verschil tussen twee groepen."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1)**2 + (len(b) - 1) * b.std(ddof=1)**2)
        / (len(a) + len(b) - 2)
    )
    return float((b.mean() - a.mean()) / (pooled_std + 1e-10))


def count_events_fast(freq_shift: np.ndarray, threshold: float,
                      sfreq: float = DS_FREQ) -> int:
    """Vectorized event teller op downgesampled signaal."""
    min_s   = max(1, int(MIN_DUR_SEC   * sfreq))
    max_s   = int(MAX_DUR_SEC          * sfreq)
    merge_s = max(1, int(MERGE_GAP_SEC * sfreq))

    active  = (freq_shift > threshold).astype(np.int8)
    padded  = np.concatenate([[0], active, [0]])
    diff    = np.diff(padded)
    starts  = np.where(diff ==  1)[0]
    ends    = np.where(diff == -1)[0]

    if len(starts) == 0:
        return 0

    mask         = ((ends - starts) >= min_s) & ((ends - starts) <= max_s)
    starts, ends = starts[mask], ends[mask]

    if len(starts) == 0:
        return 0

    merged_s = [starts[0]]
    merged_e = [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_e[-1] < merge_s:
            merged_e[-1] = e
        else:
            merged_s.append(s)
            merged_e.append(e)

    return sum(1 for s, e in zip(merged_s, merged_e) if min_s <= (e - s) <= max_s)


def build_shift_table(ridges: dict) -> dict:
    """Voorberekent freq_shift voor alle (subject, nacht, rolling) combinaties."""
    result = {}
    for sid, nights in tqdm(ridges.items(), desc="  subjects", leave=False):
        result[sid] = [
            [compute_freq_shift(night, rolling, sfreq=DS_FREQ)
             for rolling in ROLLING_GRID]
            for night in nights
        ]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stap 2a — Monte Carlo op events per uur (originele versie)
# ══════════════════════════════════════════════════════════════════════════════

def run_step2_monte_carlo() -> tuple[dict, pd.DataFrame]:
    """
    2D Monte Carlo over (ROLLING_SEC, THRESHOLD).
    Uitkomstmaat: events per uur per subject.
    Score: abs(Cohen's d) × consistentie (fractie bootstraps met |d| > 0.5)
    """
    print("\n" + "="*60)
    print("STAP 2a — Monte Carlo op events per uur")
    print(f"Iteraties:   {N_MONTE_CARLO}")
    print(f"ROLLING:     {ROLLING_GRID} sec")
    print(f"THRESHOLD:   [{THRESHOLD_MIN:.1f}, {THRESHOLD_MAX:.1f}] Hz")
    print(f"Bootstrap:   {N_BOOTSTRAP}× subsample groep A → groep B grootte")
    print("="*60)

    ds_step = int(SFREQ / DS_FREQ)

    def load_ds(sid):
        return [r[::ds_step].astype(np.float32) for r in load_cached_nights(sid)]

    ridges_a = {sid: load_ds(sid) for sid in GROUP_A_IDS}
    ridges_b = {sid: load_ds(sid) for sid in GROUP_B_IDS}
    ridges_a = {k: v for k, v in ridges_a.items() if v}
    ridges_b = {k: v for k, v in ridges_b.items() if v}

    print(f"\nGroep A: {len(ridges_a)}/{len(GROUP_A_IDS)} subjects, "
          f"{sum(len(v) for v in ridges_a.values())} nachten")
    print(f"Groep B: {len(ridges_b)}/{len(GROUP_B_IDS)} subjects, "
          f"{sum(len(v) for v in ridges_b.values())} nachten")

    if len(ridges_a) < 4 or len(ridges_b) < 2:
        raise RuntimeError("Te weinig gecachede data. Draai eerst Stap 1.")

    print("\nFreq_shift voorberekenen...")
    shifts_a = build_shift_table(ridges_a)
    shifts_b = build_shift_table(ridges_b)

    def eph_subject(sid, shifts, r_idx, threshold):
        vals = []
        for night_shifts in shifts[sid]:
            shift = night_shifts[r_idx]
            n_ev  = count_events_fast(shift, threshold, sfreq=DS_FREQ)
            hours = len(shift) / DS_FREQ / 3600.0
            vals.append(n_ev / max(hours, 0.1))
        return float(np.mean(vals))

    sids_a = list(shifts_a.keys())
    sids_b = list(shifts_b.keys())
    n_b    = len(sids_b)
    results = []

    for _ in tqdm(range(N_MONTE_CARLO), desc="MC iteraties"):
        r_idx     = np.random.randint(len(ROLLING_GRID))
        threshold = np.random.uniform(THRESHOLD_MIN, THRESHOLD_MAX)

        eph_b = np.array([eph_subject(sid, shifts_b, r_idx, threshold)
                          for sid in sids_b])

        d_samples = []
        for _ in range(N_BOOTSTRAP):
            sample     = np.random.choice(sids_a, size=n_b, replace=False)
            eph_a_boot = np.array([eph_subject(sid, shifts_a, r_idx, threshold)
                                   for sid in sample])
            d_samples.append(cohens_d(eph_a_boot, eph_b))

        mean_d      = float(np.mean(d_samples))
        consistency = float(np.mean([abs(d) > 0.5 for d in d_samples]))
        score       = abs(mean_d) * consistency

        eph_a_all = np.mean([eph_subject(sid, shifts_a, r_idx, threshold)
                             for sid in sids_a])

        results.append({
            'rolling_sec': ROLLING_GRID[r_idx],
            'threshold':   threshold,
            'cohens_d':    mean_d,
            'consistency': consistency,
            'mean_eph_a':  float(eph_a_all),
            'mean_eph_b':  float(eph_b.mean()),
            'score':       score,
        })

    df = pd.DataFrame(results)
    df.to_csv(CACHE_DIR / "mc_results_eph.csv", index=False)

    beste     = df.loc[df['score'].idxmax()]
    print(f"\n{'='*60}")
    print("RESULTAAT (events per uur)")
    print(f"  ROLLING_SEC:   {beste['rolling_sec']:.0f} sec")
    print(f"  THRESHOLD:     {beste['threshold']:.3f} Hz")
    print(f"  Cohen's d:     {beste['cohens_d']:.3f}")
    print(f"  Consistentie:  {beste['consistency']:.0%}")
    print(f"  EPH groep A:   {beste['mean_eph_a']:.1f}")
    print(f"  EPH groep B:   {beste['mean_eph_b']:.1f}")
    print(f"{'='*60}")

    optimal = {
        'ROLLING_SEC':            int(beste['rolling_sec']),
        'AROUSAL_FREQ_THRESHOLD': round(float(beste['threshold']), 3),
        'cohens_d':               round(float(beste['cohens_d']), 3),
        'consistency':            round(float(beste['consistency']), 2),
        'mean_eph_group_a':       round(float(beste['mean_eph_a']), 1),
        'mean_eph_group_b':       round(float(beste['mean_eph_b']), 1),
        'n_mc_iterations':        N_MONTE_CARLO,
        'n_bootstrap':            N_BOOTSTRAP,
    }
    with open(CACHE_DIR / "optimal_params_eph.json", 'w') as f:
        json.dump(optimal, f, indent=2)
    print(f"Opgeslagen: optimal_params_eph.json")

    return optimal, df


# ══════════════════════════════════════════════════════════════════════════════
# Stap 2b — Monte Carlo op event-kenmerken (nieuwe versie)
# ══════════════════════════════════════════════════════════════════════════════

def extract_event_features_fast(freq_shift_ds: np.ndarray,
                                 ridge_freq_ds: np.ndarray,
                                 threshold: float,
                                 sfreq: float = DS_FREQ) -> pd.DataFrame:
    """
    Detecteert events en berekent kenmerken per event op downgesampled signaal.
    Geeft DataFrame met één rij per event (leeg als geen events gevonden).
    """
    min_s   = max(1, int(MIN_DUR_SEC   * sfreq))
    max_s   = int(MAX_DUR_SEC          * sfreq)
    merge_s = max(1, int(MERGE_GAP_SEC * sfreq))

    active = (freq_shift_ds > threshold).astype(np.int8)
    padded = np.concatenate([[0], active, [0]])
    diff   = np.diff(padded)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    if len(starts) == 0:
        return pd.DataFrame()

    mask         = ((ends - starts) >= min_s) & ((ends - starts) <= max_s)
    starts, ends = starts[mask], ends[mask]

    if len(starts) == 0:
        return pd.DataFrame()

    merged_s, merged_e = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_e[-1] < merge_s:
            merged_e[-1] = e
        else:
            merged_s.append(s)
            merged_e.append(e)

    records = []
    for s, e in zip(merged_s, merged_e):
        dur = e - s
        if not (min_s <= dur <= max_s):
            continue
        records.append({
            'duration':        dur / sfreq,
            'mean_ridge_freq': float(ridge_freq_ds[s:e].mean()),
            'peak_ridge_freq': float(ridge_freq_ds[s:e].max()),
            'mean_freq_shift': float(freq_shift_ds[s:e].mean()),
            'peak_freq_shift': float(freq_shift_ds[s:e].max()),
        })

    return pd.DataFrame(records)


def subject_event_means(sid: str, shifts: dict, ridges_ds: dict,
                         r_idx: int, threshold: float) -> dict | None:
    """
    Berekent gemiddelde event-kenmerken over alle nachten van één subject.
    Geeft None als er geen events gevonden worden.
    """
    all_events = []
    for night_shifts, ridge_ds in zip(shifts[sid], ridges_ds[sid]):
        df = extract_event_features_fast(night_shifts[r_idx], ridge_ds, threshold)
        if not df.empty:
            all_events.append(df)

    if not all_events:
        return None

    combined = pd.concat(all_events, ignore_index=True)
    return {
        'duration':        combined['duration'].mean(),
        'mean_ridge_freq': combined['mean_ridge_freq'].mean(),
        'mean_freq_shift': combined['mean_freq_shift'].mean(),
        'n_events':        len(combined),
    }


def run_step2_monte_carlo_features() -> tuple[dict, pd.DataFrame]:
    """
    2D Monte Carlo over (ROLLING_SEC, THRESHOLD).
    Uitkomstmaat: event-kenmerken per subject (ridge_freq, freq_shift, duur).
    Score: gemiddelde abs(Cohen's d) × consistentie over drie kenmerken.
    Consistentie = max(fractie bootstraps positief, fractie negatief) > 0.3.
    """
    print("\n" + "="*60)
    print("STAP 2b — Monte Carlo op event-kenmerken")
    print(f"Iteraties:   {N_MONTE_CARLO}")
    print(f"Kenmerken:   ridge_freq, freq_shift, duur per event")
    print(f"ROLLING:     {ROLLING_GRID} sec")
    print(f"THRESHOLD:   [{THRESHOLD_MIN:.1f}, {THRESHOLD_MAX:.1f}] Hz")
    print(f"Bootstrap:   {N_BOOTSTRAP}× subsample groep A → groep B grootte")
    print("="*60)

    ds_step = int(SFREQ / DS_FREQ)

    def load_ds(sid):
        return [r[::ds_step].astype(np.float32) for r in load_cached_nights(sid)]

    ridges_a = {sid: load_ds(sid) for sid in GROUP_A_IDS}
    ridges_b = {sid: load_ds(sid) for sid in GROUP_B_IDS}
    ridges_a = {k: v for k, v in ridges_a.items() if v}
    ridges_b = {k: v for k, v in ridges_b.items() if v}

    print(f"\nGroep A: {len(ridges_a)}/{len(GROUP_A_IDS)} subjects, "
          f"{sum(len(v) for v in ridges_a.values())} nachten")
    print(f"Groep B: {len(ridges_b)}/{len(GROUP_B_IDS)} subjects, "
          f"{sum(len(v) for v in ridges_b.values())} nachten")

    if len(ridges_a) < 4 or len(ridges_b) < 2:
        raise RuntimeError("Te weinig gecachede data. Draai eerst Stap 1.")

    print("\nFreq_shift voorberekenen...")
    shifts_a = build_shift_table(ridges_a)
    shifts_b = build_shift_table(ridges_b)

    sids_a        = list(shifts_a.keys())
    sids_b        = list(shifts_b.keys())
    n_b           = len(sids_b)
    FEATURE_COLS  = ['mean_ridge_freq', 'mean_freq_shift', 'duration']
    results       = []

    for _ in tqdm(range(N_MONTE_CARLO), desc="MC iteraties"):
        r_idx     = np.random.randint(len(ROLLING_GRID))
        threshold = np.random.uniform(THRESHOLD_MIN, THRESHOLD_MAX)

        # groep B kenmerken
        means_b = []
        for sid in sids_b:
            m = subject_event_means(sid, shifts_b, ridges_b, r_idx, threshold)
            if m:
                means_b.append(m)

        if len(means_b) < 2:
            continue

        df_b = pd.DataFrame(means_b)

        # bootstrap groep A
        d_per_feature = {col: [] for col in FEATURE_COLS}

        for _ in range(N_BOOTSTRAP):
            sample       = np.random.choice(sids_a, size=n_b, replace=False)
            means_a_boot = []
            for sid in sample:
                m = subject_event_means(sid, shifts_a, ridges_a, r_idx, threshold)
                if m:
                    means_a_boot.append(m)

            if len(means_a_boot) < 2:
                continue

            df_a = pd.DataFrame(means_a_boot)
            for col in FEATURE_COLS:
                if col in df_a.columns and col in df_b.columns:
                    d_per_feature[col].append(
                        cohens_d(df_a[col].values, df_b[col].values)
                    )

        # score per kenmerk: abs(d) × consistentie in één richting
        feature_scores = []
        for col in FEATURE_COLS:
            ds = d_per_feature[col]
            if not ds:
                continue
            mean_d  = np.mean(ds)
            consist = max(
                np.mean([d >  0.3 for d in ds]),   # consistent positief
                np.mean([d < -0.3 for d in ds])    # consistent negatief
            )
            feature_scores.append(abs(mean_d) * consist)

        if not feature_scores:
            continue

        score = float(np.mean(feature_scores))

        results.append({
            'rolling_sec':       ROLLING_GRID[r_idx],
            'threshold':         threshold,
            'score':             score,
            'mean_d_ridge_freq': float(np.mean(d_per_feature['mean_ridge_freq']))
                                 if d_per_feature['mean_ridge_freq'] else 0.0,
            'mean_d_freq_shift': float(np.mean(d_per_feature['mean_freq_shift']))
                                 if d_per_feature['mean_freq_shift'] else 0.0,
            'mean_d_duration':   float(np.mean(d_per_feature['duration']))
                                 if d_per_feature['duration'] else 0.0,
            'mean_ridge_b':      float(df_b['mean_ridge_freq'].mean())
                                 if 'mean_ridge_freq' in df_b.columns else 0.0,
            'mean_shift_b':      float(df_b['mean_freq_shift'].mean())
                                 if 'mean_freq_shift' in df_b.columns else 0.0,
        })

    if not results:
        print("Geen resultaten — verlaag THRESHOLD_MIN of controleer de data.")
        return None, pd.DataFrame()

    df = pd.DataFrame(results)
    df.to_csv(CACHE_DIR / "mc_results_features.csv", index=False)

    beste     = df.loc[df['score'].idxmax()]
    print(f"\n{'='*60}")
    print("RESULTAAT (event-kenmerken)")
    print(f"  ROLLING_SEC:          {beste['rolling_sec']:.0f} sec")
    print(f"  THRESHOLD:            {beste['threshold']:.3f} Hz")
    print(f"  Score:                {beste['score']:.3f}")
    print(f"  Cohen's d ridge_freq: {beste['mean_d_ridge_freq']:.3f}")
    print(f"  Cohen's d freq_shift: {beste['mean_d_freq_shift']:.3f}")
    print(f"  Cohen's d duration:   {beste['mean_d_duration']:.3f}")
    print(f"{'='*60}")

    optimal = {
        'ROLLING_SEC':             int(beste['rolling_sec']),
        'AROUSAL_FREQ_THRESHOLD':  round(float(beste['threshold']), 3),
        'score':                   round(float(beste['score']), 3),
        'cohens_d_ridge_freq':     round(float(beste['mean_d_ridge_freq']), 3),
        'cohens_d_freq_shift':     round(float(beste['mean_d_freq_shift']), 3),
        'cohens_d_duration':       round(float(beste['mean_d_duration']), 3),
        'n_mc_iterations':         N_MONTE_CARLO,
        'n_bootstrap':             N_BOOTSTRAP,
    }
    with open(CACHE_DIR / "optimal_params_features.json", 'w') as f:
        json.dump(optimal, f, indent=2)
    print(f"Opgeslagen: optimal_params_features.json")

    return optimal, df


# ══════════════════════════════════════════════════════════════════════════════
# Stap 3 — Visualisatie
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame, optimal: dict, score_col: str = 'score'):
    """Maakt drie plots: threshold vs score, score per rolling venster, groepsseparatie."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rolling_vals = sorted(df['rolling_sec'].unique())
    colors       = plt.cm.tab10(np.linspace(0, 1, len(rolling_vals)))

    # plot 1: threshold vs score
    ax = axes[0]
    for rv, col in zip(rolling_vals, colors):
        sub = df[df['rolling_sec'] == rv]
        ax.scatter(sub['threshold'], sub[score_col], color=col, s=20,
                   alpha=0.6, label=f"{int(rv)}s")
    ax.axvline(optimal['AROUSAL_FREQ_THRESHOLD'], color='red', lw=1.5,
               linestyle='--',
               label=f"optimum ({optimal['AROUSAL_FREQ_THRESHOLD']} Hz)")
    ax.set_xlabel("THRESHOLD (Hz)")
    ax.set_ylabel("Score")
    ax.set_title("Score per threshold\n(kleur = ROLLING_SEC)")
    ax.legend(fontsize=7, ncol=2)

    # plot 2: score per rolling venster
    ax2    = axes[1]
    grouped = [df[df['rolling_sec'] == rv][score_col].values
               for rv in rolling_vals]
    bp = ax2.boxplot(grouped, labels=[str(int(r)) for r in rolling_vals],
                     patch_artist=True)
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    if optimal['ROLLING_SEC'] in rolling_vals:
        opt_x = rolling_vals.index(optimal['ROLLING_SEC']) + 1
        ax2.axvline(opt_x, color='red', lw=1.5, linestyle='--', label='optimum')
    ax2.set_xlabel("ROLLING_SEC (seconden)")
    ax2.set_ylabel("Score verdeling")
    ax2.set_title("Score per rolling venster")
    ax2.legend()

    # plot 3: Cohen's d per kenmerk (alleen voor features versie)
    ax3 = axes[2]
    if 'mean_d_ridge_freq' in df.columns:
        for col, label, color in zip(
            ['mean_d_ridge_freq', 'mean_d_freq_shift', 'mean_d_duration'],
            ['Ridge freq', 'Freq shift', 'Duur'],
            ['steelblue', 'darkorange', 'seagreen']
        ):
            ax3.scatter(df['threshold'], df[col], s=10, alpha=0.4,
                        color=color, label=label)
        ax3.axhline(0, color='black', lw=0.8, linestyle='--')
        ax3.axvline(optimal['AROUSAL_FREQ_THRESHOLD'], color='red', lw=1.5,
                    linestyle='--')
        ax3.set_xlabel("THRESHOLD (Hz)")
        ax3.set_ylabel("Cohen's d")
        ax3.set_title("Cohen's d per kenmerk")
        ax3.legend(fontsize=8)
    elif 'mean_eph_a' in df.columns:
        sc = ax3.scatter(df['mean_eph_a'], df['mean_eph_b'],
                         c=df[score_col], cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(sc, ax=ax3, label="Score")
        ax3.set_xlabel("Events/uur groep A")
        ax3.set_ylabel("Events/uur groep B")
        ax3.set_title("Groepsseparatie events/uur")

    plt.tight_layout()
    out = CACHE_DIR / "mc_parameter_search.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot opgeslagen: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Stap 4 — Top-N analyse
# ══════════════════════════════════════════════════════════════════════════════

def print_top_results(df: pd.DataFrame, n: int = 10):
    """Print de top-N beste parametersets."""
    score_cols = [c for c in ['score', 'cohens_d', 'consistency',
                               'mean_d_ridge_freq', 'mean_d_freq_shift',
                               'mean_d_duration', 'mean_eph_a', 'mean_eph_b']
                  if c in df.columns]

    print(f"\nTop {n} parametersets (gesorteerd op score):")
    print("-" * 75)
    top = df.nlargest(n, 'score')[['rolling_sec', 'threshold'] + score_cols].round(3)
    print(top.to_string(index=False))
    print("-" * 75)
    print(f"\nBereik in top-{n}:")
    print(f"  ROLLING_SEC: {sorted(top['rolling_sec'].unique().tolist())}")
    print(f"  THRESHOLD:   {top['threshold'].min():.3f} – {top['threshold'].max():.3f} Hz")
    print(f"\nTip: lage consistentie (<60%) = parameters gevoelig voor subjectsampling "
          f"→ verhoog N_MONTE_CARLO of gebruik meer subjects.")


# ══════════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── stap 1: cache ridge data (wordt overgeslagen als al gedaan) ──
    run_step1_cache()

    # ── stap 2: Monte Carlo op event-kenmerken ──
    optimal, mc_df = run_step2_monte_carlo_features()

    if optimal is None:
        print("Monte Carlo mislukt. Controleer de cache en groeps-IDs.")
        exit(1)

    # ── stap 3: top resultaten ──
    print_top_results(mc_df, n=30)

    # ── stap 4: visualisatie ──
    plot_results(mc_df, optimal)

    # ── eindresultaat ──
    print("\n" + "="*60)
    print("GEBRUIK DEZE WAARDEN IN microarousal_pipeline_MANUAL.py:")
    print(f"  ROLLING_SEC            = {optimal['ROLLING_SEC']}")
    print(f"  AROUSAL_FREQ_THRESHOLD = {optimal['AROUSAL_FREQ_THRESHOLD']}")
    print("="*60)