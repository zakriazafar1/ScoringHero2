"""
Microarousal Detection Pipeline — volledig geïntegreerd met Monte Carlo
=======================================================================
Volgorde:
  FASE A: bereken CWT ridge data voor alle nachten (geen threshold nodig)
  FASE B: Monte Carlo → vind optimale threshold op basis van groepsverschillen
  FASE C: detecteer events met optimale threshold → master_events.csv

Gebruik:
  python microarousal_pipeline_full.py
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


# ══════════════════════════════════════════════════════════════════════════════
# Configuratie — pas hier aan, nergens anders
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
OUTPUT_DIR = Path(r"C:\Users\zafar\Documents\bnbd_output5")
EXCEL_PATH = Path(r"C:\Users\zafar\Documents\BNBD data availability\BNBD_DATA_OVERVIEW_ZZ.xlsx")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_NIGHTS    = 2     # None = alle nachten, bijv. 10 voor pilot
N_MONTE_CARLO = 100    # aantal Monte Carlo iteraties

EEG_CH = ['EEG L psg-lp', 'EEG R psg-lp']
EMG_CH = ['EEG L psg-emg', 'EEG R psg-emg']
MOV_CH = ['dX', 'dY', 'dZ']
ALL_CH = EEG_CH + EMG_CH + MOV_CH

SFREQ       = 256.0
FREQS       = np.arange(0.5, 35.5, 0.5)   # 70 frequenties: 0.5–35 Hz
ROLLING_SEC = 60.0                          # baseline venster in seconden

BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 35.0),
}

# event detectie parameters
AROUSAL_MIN_DUR = 2.0    # minimale eventduur in seconden
AROUSAL_MAX_DUR = 30.0   # maximale eventduur in seconden
MERGE_GAP_SEC   = 1.0    # merge events met gap kleiner dan dit


# ══════════════════════════════════════════════════════════════════════════════
# Signaalverwerking — laden, filteren, CWT, baseline
# ══════════════════════════════════════════════════════════════════════════════

def load_night(edf_path):
    """Laadt EDF en past filters toe."""
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
    return raw


def preprocess_signals(raw):
    """Haalt alle signalen op als numpy arrays."""
    return {ch: raw.get_data(picks=ch)[0] for ch in ALL_CH}


def compute_morlet_tf_streaming(signal, srate, freqs, bands, n_cycles=None):
    """
    Berekent CWT zonder de volledige power matrix op te slaan.
    Geheugen: ~168 MB i.p.v. ~1.93 GiB voor 8 uur data.
    Geeft: ridge_freq, ridge_power, band_mean (dict per band)
    """
    freqs = np.asarray(freqs)
    if n_cycles is None:
        n_cycles_arr = np.maximum(3.0, freqs / 2.0)
    elif np.isscalar(n_cycles):
        n_cycles_arr = np.full(len(freqs), float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)

    n_samples  = len(signal)
    signal     = signal - np.mean(signal)
    signal_fft = np.fft.fft(signal)
    fft_freqs  = np.fft.fftfreq(n_samples, d=1.0 / srate)

    ridge_max  = np.full(n_samples, -np.inf, dtype=np.float32)
    ridge_idx  = np.zeros(n_samples, dtype=np.uint8)
    band_accum = {name: np.zeros(n_samples, dtype=np.float32) for name in bands}
    band_count = {name: 0 for name in bands}

    for i, freq in enumerate(tqdm(freqs, desc="CWT", leave=False)):
        sigma_f     = freq / n_cycles_arr[i]
        wavelet_fft = np.exp(-0.5 * ((fft_freqs - freq) / sigma_f) ** 2)
        analytic    = np.fft.ifft(signal_fft * wavelet_fft)
        power_row   = (np.abs(analytic) ** 2).astype(np.float32)

        better = power_row > ridge_max
        ridge_max[better] = power_row[better]
        ridge_idx[better] = i

        for name, (lo, hi) in bands.items():
            if lo <= freq <= hi:
                band_accum[name] += power_row
                band_count[name] += 1

    ridge_freq  = freqs[ridge_idx]
    ridge_power = ridge_max
    band_mean   = {
        name: band_accum[name] / band_count[name] if band_count[name] > 0
        else np.zeros(n_samples, dtype=np.float32)
        for name in bands
    }
    return ridge_freq, ridge_power, band_mean


def compute_freq_shift(ridge_freq, srate, baseline_sec=ROLLING_SEC):
    """
    Berekent hoe ver de ridge omhoog springt t.o.v. lokale baseline.
    Gebruikt pandas rolling mediaan — snel en geheugenefficiënt.
    """
    window   = int(baseline_sec * srate)
    series   = pd.Series(ridge_freq)
    baseline = series.rolling(window=window, min_periods=1).median().values
    return ridge_freq - baseline, baseline


# ══════════════════════════════════════════════════════════════════════════════
# FASE A — sla ridge data op voor alle nachten (zonder threshold)
# ══════════════════════════════════════════════════════════════════════════════

def save_ridge_data(edf_path, subject_id, night_id):
    """
    Berekent CWT en slaat ridge_freq + freq_shift op per seconde.
    Geen threshold, geen event detectie — puur ruwe signaaldata.
    Dit is de input voor de Monte Carlo in Fase B.
    """
    cache_path = OUTPUT_DIR / f"{subject_id}_{night_id}_ridge.parquet"
    if cache_path.exists():
        print(f"  Al berekend, overgeslagen: {cache_path.name}")
        return

    print("  [1/3] Laden en filteren...")
    raw     = load_night(edf_path)
    signals = preprocess_signals(raw)
    n_samp  = len(signals[EEG_CH[0]])
    del raw

    print(f"  [2/3] CWT berekenen ({n_samp/SFREQ/3600:.1f} uur)...")
    eeg_avg = (signals[EEG_CH[0]] + signals[EEG_CH[1]]) / 2.0
    ridge_freq, ridge_power, band_mean = compute_morlet_tf_streaming(
        eeg_avg, SFREQ, FREQS, BANDS
    )

    print("  [3/3] Baseline berekenen en opslaan...")
    freq_shift, _ = compute_freq_shift(ridge_freq, SFREQ, ROLLING_SEC)

    emg_avg = (np.abs(signals[EMG_CH[0]]) + np.abs(signals[EMG_CH[1]])) / 2.0
    acc     = np.sqrt(signals[MOV_CH[0]]**2 +
                      signals[MOV_CH[1]]**2 +
                      signals[MOV_CH[2]]**2)

    # downsample naar 1 waarde per seconde
    step = int(SFREQ)
    df = pd.DataFrame({
        'subject_id':  subject_id,
        'night_id':    night_id,
        'time_sec':    np.arange(0, n_samp, step) / SFREQ,
        'ridge_freq':  ridge_freq[::step].astype(np.float32),
        'freq_shift':  freq_shift[::step].astype(np.float32),
        'ridge_power': ridge_power[::step].astype(np.float32),
        'emg':         emg_avg[::step].astype(np.float32),
        'acc':         acc[::step].astype(np.float32),
    })
    df.to_parquet(cache_path, index=False)
    print(f"  Opgeslagen: {cache_path.name}")


def run_fase_a():
    """Verwerkt alle nachten en slaat ridge data op als parquet."""
    print("\n" + "="*60)
    print("FASE A — Ridge data berekenen voor alle nachten")
    print("="*60)

    all_edf = sorted(f for f in BASE_DIR.rglob("*_psg.edf") if "_T0_" in f.name)
    print(f"Gevonden: {len(all_edf)} PSG bestanden (T0)")
    if MAX_NIGHTS:
        all_edf = all_edf[:MAX_NIGHTS]
        print(f"Beperkt tot: {MAX_NIGHTS} nachten (MAX_NIGHTS)")

    for i, edf_path in enumerate(all_edf):
        stem       = edf_path.stem.replace("_psg", "")
        parts      = stem.split("_")
        subject_id = f"bnbd_nsr_{parts[2]}"
        night_id   = f"{parts[3]}_{parts[4]}"
        print(f"\n[{i+1}/{len(all_edf)}] {subject_id} / {night_id}")
        try:
            save_ridge_data(edf_path, subject_id, night_id)
        except Exception as e:
            print(f"  FOUT: {e}")

    n_done = len(list(OUTPUT_DIR.glob("*_ridge.parquet")))
    print(f"\nFase A klaar — {n_done} ridge bestanden opgeslagen")


# ══════════════════════════════════════════════════════════════════════════════
# FASE B — Monte Carlo threshold optimalisatie
# ══════════════════════════════════════════════════════════════════════════════

def load_group_map():
    """Laadt groepslabels uit Excel. Geeft dict: subject_id → NSR/Prezens/SAV"""
    overview = pd.read_excel(EXCEL_PATH)
    overview['subject_id'] = overview['subject_number'].apply(
        lambda x: f"bnbd_nsr_{int(x):05d}"
    )
    return (overview.drop_duplicates('subject_id')
                    .set_index('subject_id')['study']
                    .to_dict())


def run_fase_b():
    """
    Monte Carlo threshold optimalisatie.
    Zoekt de threshold die NSR / Prezens / SAV het best van elkaar scheidt op:
      - events per uur
      - gemiddelde freq_shift
      - gemiddelde ridge frequentie
    """
    print("\n" + "="*60)
    print("FASE B — Monte Carlo threshold optimalisatie")
    print("="*60)

    group_map  = load_group_map()
    all_ridge  = sorted(OUTPUT_DIR.glob("*_ridge.parquet"))
    print(f"Ridge bestanden gevonden: {len(all_ridge)}")

    if not all_ridge:
        print("Geen ridge bestanden gevonden. Draai eerst Fase A.")
        return None

    dfs = []
    for f in all_ridge:
        df  = pd.read_parquet(f)
        sid = df['subject_id'].iloc[0]
        grp = group_map.get(sid)
        if grp:
            df['group'] = grp
            dfs.append(df)

    if not dfs:
        print("Geen data met groepslabels gevonden.")
        return None

    master_ridge = pd.concat(dfs, ignore_index=True)
    print(f"Totaal: {len(master_ridge)} samples")
    print(f"Groepen: {master_ridge['group'].value_counts().to_dict()}")

    print(f"\n{N_MONTE_CARLO} iteraties draaien...")
    results = []

    for _ in tqdm(range(N_MONTE_CARLO), desc="Monte Carlo"):
        threshold = np.random.uniform(0.1, 5.0)

        records = []
        for (sid, nid, grp), g in master_ridge.groupby(
                ['subject_id', 'night_id', 'group']):
            above     = (g['freq_shift'] > threshold).sum()
            nacht_uur = len(g) / 3600.0
            records.append({
                'group':           grp,
                'events_per_uur':  above / max(nacht_uur, 0.1),
                'mean_freq_shift': float(g['freq_shift'].mean()),
                'mean_ridge_freq': float(g['ridge_freq'].mean()),
            })

        metrics = pd.DataFrame(records)
        groepen  = [metrics[metrics['group'] == g]
                    for g in ['NSR', 'Prezens', 'SAV']]
        groepen  = [g for g in groepen if len(g) >= 2]
        if len(groepen) < 2:
            continue

        scores = []
        for col in ['events_per_uur', 'mean_freq_shift', 'mean_ridge_freq']:
            try:
                stat, _ = stats.kruskal(*[g[col].values for g in groepen])
                scores.append(stat)
            except Exception:
                scores.append(0.0)

        results.append({
            'threshold':      threshold,
            'kruskal_events': scores[0],
            'kruskal_shift':  scores[1],
            'kruskal_ridge':  scores[2],
            'combined_score': float(np.mean(scores)),
        })

    if not results:
        print("Monte Carlo leverde geen resultaten op.")
        return None

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "monte_carlo_results.csv", index=False)

    beste_idx       = results_df['combined_score'].idxmax()
    beste_threshold = float(results_df.loc[beste_idx, 'threshold'])
    beste_score     = float(results_df.loc[beste_idx, 'combined_score'])

    print(f"\n{'='*60}")
    print(f"RESULTAAT MONTE CARLO")
    print(f"Beste threshold:  {beste_threshold:.3f} Hz")
    print(f"Scheiding score:  {beste_score:.3f}")

    # groepsverschillen tonen bij beste threshold
    records = []
    for (sid, nid, grp), g in master_ridge.groupby(
            ['subject_id', 'night_id', 'group']):
        above     = (g['freq_shift'] > beste_threshold).sum()
        nacht_uur = len(g) / 3600.0
        records.append({
            'group':           grp,
            'events_per_uur':  above / max(nacht_uur, 0.1),
            'mean_freq_shift': float(g['freq_shift'].mean()),
            'mean_ridge_freq': float(g['ridge_freq'].mean()),
        })
    print("\nGroepsverschillen bij beste threshold:")
    print(pd.DataFrame(records).groupby('group')[
        ['events_per_uur', 'mean_freq_shift', 'mean_ridge_freq']
    ].mean().round(2).to_string())

    # sla threshold op
    with open(OUTPUT_DIR / "optimal_threshold.json", 'w') as f:
        json.dump({'threshold': beste_threshold, 'score': beste_score}, f)
    print(f"\nThreshold opgeslagen: optimal_threshold.json")

    return beste_threshold


# ══════════════════════════════════════════════════════════════════════════════
# FASE C — event detectie met optimale threshold
# ══════════════════════════════════════════════════════════════════════════════

def detect_events(ridge_freq, ridge_power, freq_shift, srate, threshold):
    """Detecteert kandidaat-events op basis van ridge-stijging."""
    min_s   = int(AROUSAL_MIN_DUR * srate)
    max_s   = int(AROUSAL_MAX_DUR * srate)
    merge_s = int(MERGE_GAP_SEC   * srate)

    active = freq_shift > threshold

    raw_events = []
    in_ev, start = False, 0
    for i, flag in enumerate(active):
        if flag and not in_ev:
            start, in_ev = i, True
        elif not flag and in_ev:
            in_ev = False
            if min_s <= (i - start) <= max_s:
                raw_events.append((start, i))
    if in_ev:
        dur = len(active) - start
        if min_s <= dur <= max_s:
            raw_events.append((start, len(active)))

    merged = []
    for s, e in raw_events:
        if merged and (s - merged[-1][1]) < merge_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if min_s <= (e - s) <= max_s]


def extract_event_features(start, end, band_mean, ridge_freq,
                            ridge_power, freq_shift, emg_avg, acc, srate):
    """Berekent alle features voor één event."""
    delta = float(band_mean['delta'][start:end].mean())
    theta = float(band_mean['theta'][start:end].mean())
    alpha = float(band_mean['alpha'][start:end].mean())
    beta  = float(band_mean['beta' ][start:end].mean())
    total = delta + theta + alpha + beta + 1e-10

    seg   = ridge_freq[start:end]
    slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) if len(seg) > 2 else 0.0

    return {
        'start_sec':       start / srate,
        'end_sec':         end   / srate,
        'duration':        (end - start) / srate,
        'mean_ridge_freq': float(seg.mean()),
        'peak_ridge_freq': float(seg.max()),
        'freq_shift':      float(freq_shift[start:end].mean()),
        'peak_power':      float(ridge_power[start:end].max()),
        'mean_power':      float(ridge_power[start:end].mean()),
        'ridge_slope':     slope,
        'pow_delta':       delta / total,
        'pow_theta':       theta / total,
        'pow_alpha':       alpha / total,
        'pow_beta':        beta  / total,
        'fast_slow_ratio': (alpha + beta) / (delta + theta + 1e-10),
        'emg_mean':        float(np.mean(np.abs(emg_avg[start:end]))),
        'acc_mean':        float(np.mean(acc[start:end])),
    }


def label_artifact(row, emg_vals, acc_vals, emg_pct=90, acc_pct=90):
    """Markeert events als mogelijk artefact — gooit ze niet weg."""
    emg_thr = np.percentile(emg_vals, emg_pct)
    acc_thr = np.percentile(acc_vals, acc_pct)
    if row['emg_mean'] > emg_thr and row['acc_mean'] > acc_thr:
        return 'movement'
    elif row['emg_mean'] > emg_thr:
        return 'emg_dominant'
    elif row['acc_mean'] > acc_thr:
        return 'acc_dominant'
    return 'clean'


def process_one_night(edf_path, subject_id, night_id, threshold):
    """Verwerkt één nacht met de gegeven threshold."""
    night_out = OUTPUT_DIR / f"{subject_id}_{night_id}_events.csv"
    if night_out.exists():
        print(f"  Al verwerkt, overgeslagen: {night_out.name}")
        return pd.read_csv(night_out)

    print(f"\n{'='*60}")
    print(f"Verwerken: {subject_id} / {night_id}")

    print("  [1/5] Laden en filteren...")
    raw     = load_night(edf_path)
    signals = preprocess_signals(raw)
    n_samp  = len(signals[EEG_CH[0]])
    del raw

    print(f"  [2/5] CWT berekenen ({n_samp/SFREQ/3600:.1f} uur)...")
    eeg_avg = (signals[EEG_CH[0]] + signals[EEG_CH[1]]) / 2.0
    ridge_freq, ridge_power, band_mean = compute_morlet_tf_streaming(
        eeg_avg, SFREQ, FREQS, BANDS
    )

    print("  [3/5] Baseline berekenen...")
    freq_shift, _ = compute_freq_shift(ridge_freq, SFREQ, ROLLING_SEC)

    print(f"  [4/5] Events detecteren (threshold={threshold:.3f} Hz)...")
    events = detect_events(ridge_freq, ridge_power, freq_shift, SFREQ, threshold)
    print(f"         → {len(events)} kandidaat-events gevonden")

    if not events:
        print("  Geen events gevonden.")
        return pd.DataFrame()

    print("  [5/5] Features berekenen...")
    emg_avg = (np.abs(signals[EMG_CH[0]]) + np.abs(signals[EMG_CH[1]])) / 2.0
    acc     = np.sqrt(signals[MOV_CH[0]]**2 +
                      signals[MOV_CH[1]]**2 +
                      signals[MOV_CH[2]]**2)

    records = []
    for start, end in events:
        feat = extract_event_features(
            start, end, band_mean, ridge_freq,
            ridge_power, freq_shift, emg_avg, acc, SFREQ
        )
        feat['subject_id'] = subject_id
        feat['night_id']   = night_id
        records.append(feat)

    df = pd.DataFrame(records)
    df['artifact_label'] = df.apply(
        lambda r: label_artifact(r, df['emg_mean'].values, df['acc_mean'].values),
        axis=1
    )

    print(f"  → {len(df)} events  "
          f"({(df['artifact_label']=='clean').sum()} schoon, "
          f"{(df['artifact_label']!='clean').sum()} artefact)")

    return df


def run_fase_c(threshold):
    """Detecteert events voor alle nachten met de optimale threshold."""
    print("\n" + "="*60)
    print(f"FASE C — Event detectie (threshold = {threshold:.3f} Hz)")
    print("="*60)

    all_edf = sorted(f for f in BASE_DIR.rglob("*_psg.edf") if "_T0_" in f.name)
    if MAX_NIGHTS:
        all_edf = all_edf[:MAX_NIGHTS]
    print(f"Verwerken: {len(all_edf)} nachten")

    group_map = load_group_map()
    all_dfs   = []

    for i, edf_path in enumerate(all_edf):
        stem       = edf_path.stem.replace("_psg", "")
        parts      = stem.split("_")
        subject_id = f"bnbd_nsr_{parts[2]}"
        night_id   = f"{parts[3]}_{parts[4]}"
        print(f"\n[{i+1}/{len(all_edf)}] {subject_id} / {night_id}")

        try:
            df = process_one_night(edf_path, subject_id, night_id, threshold)
            if df.empty:
                continue
            night_out = OUTPUT_DIR / f"{subject_id}_{night_id}_events.csv"
            df.to_csv(night_out, index=False)
            all_dfs.append(df)
        except Exception as e:
            print(f"  FOUT: {e}")

    if not all_dfs:
        print("Geen events gevonden.")
        return pd.DataFrame()

    master = pd.concat(all_dfs, ignore_index=True)
    master['group'] = master['subject_id'].map(group_map)

    master_out = OUTPUT_DIR / "master_events.csv"
    master.to_csv(master_out, index=False)

    print(f"\n{'='*60}")
    print(f"KLAAR")
    print(f"Totaal events:   {len(master)}")
    print(f"Deelnemers:      {master['subject_id'].nunique()}")
    print(f"Schone events:   {(master['artifact_label']=='clean').sum()}")
    print(f"Master tabel:    {master_out}")

    print("\nEvents per groep:")
    print(master.groupby('group').agg(
        n_events   = ('duration', 'count'),
        mean_dur   = ('duration', 'mean'),
        mean_ridge = ('mean_ridge_freq', 'mean'),
        mean_shift = ('freq_shift', 'mean'),
    ).round(2).to_string())

    return master


# ══════════════════════════════════════════════════════════════════════════════
# Kwaliteitscontrole
# ══════════════════════════════════════════════════════════════════════════════

def qc_summary(master):
    """Print overzicht per nacht."""
    print("\nKwaliteitscontrole per nacht:")
    print("-" * 60)
    for (subj, night), g in master.groupby(['subject_id', 'night_id']):
        n       = len(g)
        n_clean = (g['artifact_label'] == 'clean').sum()
        dur_hr  = (g['end_sec'].max() - g['start_sec'].min()) / 3600
        eph     = n / max(dur_hr, 0.1)
        print(f"{subj} / {night}:  "
              f"{n} events ({eph:.0f}/uur)  "
              f"{100*n_clean/n:.0f}% schoon  "
              f"gem.duur {g['duration'].mean():.1f}s  "
              f"ridge {g['mean_ridge_freq'].mean():.1f}Hz")
    print("-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── FASE A: ridge data berekenen voor alle nachten ──
    run_fase_a()

    # ── FASE B: Monte Carlo threshold ──
    threshold_path = OUTPUT_DIR / "optimal_threshold.json"

    if threshold_path.exists():
        # al eerder berekend — laad direct zodat je Fase B niet opnieuw hoeft te draaien
        with open(threshold_path) as f:
            threshold = json.load(f)['threshold']
        print(f"\nThreshold geladen uit cache: {threshold:.3f} Hz")
    else:
        threshold = run_fase_b()

    if threshold is None:
        print("Monte Carlo mislukt. Controleer of Fase A klaar is.")
        exit(1)

    print(f"\nGebruikte threshold: {threshold:.3f} Hz")

    # ── FASE C: event detectie met optimale threshold ──
    master = run_fase_c(threshold)

    # ── kwaliteitscontrole ──
    if not master.empty:
        qc_summary(master)
