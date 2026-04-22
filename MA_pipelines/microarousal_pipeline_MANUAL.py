"""
Volledige geautomatiseerde microarousal detectie pipeline
=========================================================
Startpunt: jouw preprocessing code (load_night + preprocess_signals)
Eindpunt:  CSV met alle kandidaat-events + features, klaar voor clustering
"""

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
from pathlib import Path          # voor bestandspaden (werkt op Windows en Mac)
import numpy as np                # numerieke berekeningen
import pandas as pd               # data opslaan als tabel
import mne                        # EDF bestanden inladen
from tqdm import tqdm 

# ══════════════════════════════════════════════════════════════════════════════
# Configuratie — alle parameters op één plek
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIRS = [
    Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR"),
    Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\Prezens"),
    Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\SAV"),
]
output_dir = Path(r"C:\Users\zafar\Documents\bnbd_output4")
output_dir.mkdir(exist_ok=True)

MAX_PARTICIPANTS = 3      # hoeveel deelnemers je wilt verwerken (5 voor pilot)

EEG_CH = ['EEG L psg-lp', 'EEG R psg-lp']          # EEG kanalen links en rechts
EMG_CH = ['EEG L psg-emg', 'EEG R psg-emg']         # EMG kanalen (spieractiviteit)
MOV_CH = ['dX', 'dY', 'dZ']                          # bewegingskanalen (accelerometer)
ALL_CH = EEG_CH + EMG_CH + MOV_CH                    # alle kanalen samen

SFREQ     = 256.0                 # sampling rate: 256 metingen per seconde
WIN_SEC   = 1.0                   # venstergrootte voor feature extractie: 1 seconde
STEP_SEC  = 0.5                   # stap tussen vensters: 0.5 seconde (50% overlap)
WIN_SAMP  = int(WIN_SEC  * SFREQ) # venster in samples: 1.0 × 256 = 256 samples
STEP_SAMP = int(STEP_SEC * SFREQ) # stap in samples: 0.5 × 256 = 128 samples

# frequenties voor de CWT: van 0.5 Hz tot 35 Hz in stappen van 0.5 Hz
# dit geeft 70 frequenties totaal
FREQS = np.arange(0.5, 35.5, 0.5)

# EEG frequentiebanden — gebruikt voor feature extractie per eventijm
BANDS = {
    'delta': (0.5,  4.0),   # diepe slaap
    'theta': (4.0,  8.0),   # lichte slaap / drowsiness
    'alpha': (8.0,  13.0),  # ontspanning / arousal
    'beta':  (13.0, 35.0),  # activatie / alertheid
}

ROLLING_SEC            = 60.0   # baseline venster: vergelijk met de vorige 60 seconden
AROUSAL_FREQ_THRESHOLD = 0.731    # ridge moet > 1 Hz boven baseline springen = activatie
AROUSAL_MIN_DUR        = 1.0    # event moet minimaal 1 seconden duren
AROUSAL_MAX_DUR        = 20.0   # event mag maximaal 20 seconden duren


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — Laden en preprocessen (jouw bestaande code)
# ══════════════════════════════════════════════════════════════════════════════

def load_night(edf_file):
    """
    Laadt één EDF bestand en past filters toe.
    EEG: bandpass 0.5–35 Hz (verwijdert DC drift en hoge frequentie ruis)
    EMG: bandpass 10–100 Hz (spieractiviteit zichtbaar maken)
    MOV: DC removal (verwijdert offset uit accelerometer)
    """
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
    # selecteer alleen de kanalen die we nodig hebben
    raw.pick(ALL_CH)
    # laad data in geheugen
    raw.load_data(verbose=False)
    # zet om naar float64 voor nauwkeurigere berekeningen
    raw._data = raw._data.astype(np.float64)

    # filter EEG: verwijder alles onder 0.5 Hz (DC drift) en boven 35 Hz (ruis)
    raw.filter(l_freq=0.5, h_freq=35.0, picks=EEG_CH, verbose=False)
    # filter EMG: hoog-doorlaat zodat alleen spieractiviteit overblijft
    h_emg = min(100.0, SFREQ / 2 - 1)   # max = Nyquist frequentie - 1
    raw.filter(l_freq=10.0, h_freq=h_emg, picks=EMG_CH, verbose=False)
    # beweging: verwijder gemiddelde (centreer het signaal op 0)
    raw.apply_function(lambda x: x - np.mean(x), picks=MOV_CH, verbose=False)

    return raw


def preprocess_signals(raw):
    """
    Haalt alle signalen op als numpy arrays.
    Geeft een dictionary: kanaalnaam → array van metingen
    """
    signals = {}
    for ch in ALL_CH:
        # get_data geeft een 2D array (1 kanaal × n_samples), [0] pakt de eerste rij
        signals[ch] = raw.get_data(picks=ch)[0]
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2+3 — Morlet CWT + ridge extractie (streaming, geheugenefficiënt)
# ══════════════════════════════════════════════════════════════════════════════

def compute_morlet_tf_streaming(signal, srate, freqs, bands, n_cycles=None, L2normalize=False):
    """
    Berekent CWT zonder de volledige power matrix op te slaan.

    Probleem met de originele aanpak: de power matrix (n_freqs × n_samples)
    is ~1.93 GiB voor 8 uur data. NumPy's argmax heeft dan intern nog eens
    ~1.93 GiB nodig voor een transposed copy → geheugen vol.

    Oplossing: verwerk één frequentie tegelijk en bewaar alleen:
    - ridge_max:   (n_samples,) — hoogste power per tijdstip
    - ridge_idx:   (n_samples,) — welke freq-index die max had
    - band_accum:  4 × (n_samples,) — opgetelde power per band

    Totaal geheugen: ~168 MB i.p.v. ~1.93 GiB (11× minder)

    Geeft terug:
    - ridge_freq:  (n_samples,) — dominante frequentie in Hz
    - ridge_power: (n_samples,) — power op die frequentie
    - band_mean:   dict band_naam → (n_samples,) gemiddelde power per sample
    """
    freqs = np.asarray(freqs)

    if n_cycles is None:
        n_cycles_arr = np.maximum(3.0, freqs / 2.0)
    elif np.isscalar(n_cycles):
        n_cycles_arr = np.full(len(freqs), float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)

    n_samples = len(signal)
    signal    = signal - np.mean(signal)
    signal_fft = np.fft.fft(signal)
    fft_freqs  = np.fft.fftfreq(n_samples, d=1.0 / srate)

    # Ridge accumulatoren — klein: elk (n_samples,) float32/uint8
    ridge_max = np.full(n_samples, -np.inf, dtype=np.float32)
    ridge_idx = np.zeros(n_samples, dtype=np.uint8)  # max 255 freqs, hier 70

    # Per-band power accumulatoren
    band_accum = {name: np.zeros(n_samples, dtype=np.float32) for name in bands}
    band_count = {name: 0 for name in bands}

    for i, freq in enumerate(tqdm(freqs, desc="CWT", leave=False)):
        sigma_f     = freq / n_cycles_arr[i]
        wavelet_fft = np.exp(-0.5 * ((fft_freqs - freq) / sigma_f) ** 2)

        if L2normalize:
            wavelet_fft /= np.sqrt(np.sum(wavelet_fft ** 2))

        analytic  = np.fft.ifft(signal_fft * wavelet_fft)
        power_row = (np.abs(analytic) ** 2).astype(np.float32)  # (n_samples,)

        # Update ridge: vervang waar deze frequentie hoger is
        better = power_row > ridge_max
        ridge_max[better] = power_row[better]
        ridge_idx[better] = i

        # Accumuleer band power
        for name, (lo, hi) in bands.items():
            if lo <= freq <= hi:
                band_accum[name] += power_row
                band_count[name] += 1

    # Zet ridge index om naar Hz
    ridge_freq  = freqs[ridge_idx]   # (n_samples,)
    ridge_power = ridge_max          # (n_samples,)

    # Normaliseer band accumulatoren naar gemiddelde power per sample
    band_mean = {}
    for name in bands:
        if band_count[name] > 0:
            band_mean[name] = band_accum[name] / band_count[name]
        else:
            band_mean[name] = np.zeros(n_samples, dtype=np.float32)

    return ridge_freq, ridge_power, band_mean


# ══════════════════════════════════════════════════════════════════════════════
# Fase 4 — Lokale baseline en freq_shift
# ══════════════════════════════════════════════════════════════════════════════

def compute_freq_shift(ridge_freq, srate, baseline_sec=ROLLING_SEC):
    """
    Snelle vectorized versie — gebruikt pandas rolling mediaan.
    Zelfde resultaat, maar seconden in plaats van minuten.
    """
    import pandas as pd
    
    window = int(baseline_sec * srate)   # 60 sec × 256 = 15360 samples
    
    # pandas rolling mediaan is sterk geoptimaliseerd
    series   = pd.Series(ridge_freq)
    baseline = series.rolling(window=window, min_periods=1).median().values
    
    freq_shift = ridge_freq - baseline
    
    return freq_shift, baseline

# ══════════════════════════════════════════════════════════════════════════════
# Fase 5 — Kandidaat-event detectie
# ══════════════════════════════════════════════════════════════════════════════

def detect_events(ridge_freq, ridge_power, freq_shift, srate,
                  threshold=AROUSAL_FREQ_THRESHOLD,
                  min_dur=AROUSAL_MIN_DUR,
                  max_dur=AROUSAL_MAX_DUR,
                  merge_gap_sec=1.0):
    """
    Detecteert kandidaat-events op basis van ridge-stijging.

    Logica:
    1. Markeer elk tijdstip waar freq_shift > threshold als 'actief'
    2. Groepeer aaneengesloten actieve samples tot events
    3. Verwijder events die te kort of te lang zijn
    4. Merge events die vlak na elkaar komen (< 1 seconde gap)

    Geen klinische definitie — puur op basis van het signaal zelf.
    """
    min_samp   = int(min_dur    * srate)   # 3 sec × 256 = 768 samples minimum
    max_samp   = int(max_dur    * srate)   # 30 sec × 256 = 7680 samples maximum
    merge_samp = int(merge_gap_sec * srate) # 1 sec × 256 = 256 samples merge-gap

    # boolean array: True = dit tijdstip is 'actief' (ridge springt omhoog)
    active = freq_shift > threshold

    # ── stap 1: groepeer aaneengesloten actieve samples ──
    raw_events = []      # lijst van (start_sample, eind_sample) tuples
    in_event   = False   # zijn we momenteel in een event?
    start      = 0       # startpunt van huidig event

    for i, flag in enumerate(active):
        if flag and not in_event:
            # ridge gaat boven drempel: begin van nieuw event
            start    = i
            in_event = True

        elif not flag and in_event:
            # ridge gaat onder drempel: einde van event
            in_event = False
            duration = i - start   # duur in samples

            # bewaar alleen events binnen de duurgrens
            if min_samp <= duration <= max_samp:
                raw_events.append((start, i))

    # sluit eventueel nog lopend event aan het einde van het signaal
    if in_event:
        duration = len(active) - start
        if min_samp <= duration <= max_samp:
            raw_events.append((start, len(active)))

    # ── stap 2: merge events die dicht bij elkaar liggen ──
    # als twee events < 1 seconde uit elkaar liggen, zijn ze waarschijnlijk één event
    merged = []
    for s, e in raw_events:
        if merged and (s - merged[-1][1]) < merge_samp:
            # verleng het vorige event tot het einde van dit event
            merged[-1] = (merged[-1][0], e)
        else:
            # voeg toe als nieuw event
            merged.append((s, e))

    # ── stap 3: hercheck duur na mergen ──
    final_events = []
    for s, e in merged:
        duration = e - s
        if min_samp <= duration <= max_samp:
            final_events.append((s, e))

    return final_events   # lijst van (start_sample, eind_sample) tuples


# ══════════════════════════════════════════════════════════════════════════════
# Fase 6 — Feature extractie per event
# ══════════════════════════════════════════════════════════════════════════════

def extract_event_features(start, end, band_mean, ridge_freq,
                            ridge_power, freq_shift, emg_signal, acc_signal,
                            srate):
    """
    Berekent alle features voor één event.

    Gebruikt band_mean arrays (al berekend tijdens streaming CWT) voor bandpowers.
    band_mean: dict band_naam → (n_samples,) gemiddelde power per sample.

    Features:
    - timing: start, einde, duur
    - ridge: gemiddelde freq, freq_shift, peak power
    - bandpowers: delta, theta, alpha, beta (genormaliseerd)
    - ratio: fast/slow — hoe actief was het EEG?
    - EMG: gemiddelde spieractiviteit tijdens event
    - beweging: gemiddelde accelerometer magnitude
    """
    # ── bandpowers: gemiddelde over de samples van dit event ──
    delta = float(band_mean['delta'][start:end].mean())
    theta = float(band_mean['theta'][start:end].mean())
    alpha = float(band_mean['alpha'][start:end].mean())
    beta  = float(band_mean['beta' ][start:end].mean())
    total = delta + theta + alpha + beta + 1e-10  # +kleine waarde om /0 te vermijden

    # fast/slow ratio: hoe actief was het EEG?
    # hoog = meer alpha+beta dan delta+theta = activatie
    fast_slow = (alpha + beta) / (delta + theta + 1e-10)

    # ── ridge kenmerken ──
    seg_ridge = ridge_freq[start:end]    # ridge alleen tijdens dit event

    # slope: stijgt de ridge (positief) of daalt die (negatief) tijdens het event?
    # np.polyfit past een rechte lijn door de data: [helling, constante]
    if len(seg_ridge) > 2:
        slope = float(np.polyfit(np.arange(len(seg_ridge)), seg_ridge, 1)[0])
    else:
        slope = 0.0

    # ── EMG en beweging ──
    emg_mean = float(np.mean(np.abs(emg_signal[start:end])))  # abs = rectificeren
    acc_mean = float(np.mean(acc_signal[start:end]))

    # ── sla alles op als dictionary ──
    return {
        'start_sec':       start / srate,                    # starttijd in seconden
        'end_sec':         end   / srate,                    # eindtijd in seconden
        'duration':        (end - start) / srate,            # duur in seconden
        'mean_ridge_freq': float(seg_ridge.mean()),          # gem. dominante freq
        'peak_ridge_freq': float(seg_ridge.max()),           # hoogste ridge freq
        'freq_shift':      float(freq_shift[start:end].mean()), # gem. stijging boven baseline
        'peak_power':      float(ridge_power[start:end].max()),  # hoogste power
        'mean_power':      float(ridge_power[start:end].mean()), # gem. power
        'ridge_slope':     slope,                            # stijgt/daalt ridge?
        'pow_delta':       delta / total,                    # delta fractie (genorm.)
        'pow_theta':       theta / total,                    # theta fractie
        'pow_alpha':       alpha / total,                    # alpha fractie
        'pow_beta':        beta  / total,                    # beta fractie
        'fast_slow_ratio': fast_slow,                        # activatie-index
        'emg_mean':        emg_mean,                         # spieractiviteit
        'acc_mean':        acc_mean,                         # beweging
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fase 7 — Artifact labeling
# ══════════════════════════════════════════════════════════════════════════════

def label_artifact(row, emg_vals, acc_vals, emg_pct=90, acc_pct=90):
    """
    Markeert events die waarschijnlijk geen echte microarousals zijn.

    Gooit ze NIET weg — de clustering beslist later welke clusters
    artifact-clusters zijn. Zo verlies je geen informatie.

    emg_pct=90 betekent: boven het 90e percentiel van alle events = verdacht
    """
    # bereken drempels op basis van de verdeling van alle events deze nacht
    emg_thr = np.percentile(emg_vals, emg_pct)
    acc_thr = np.percentile(acc_vals, acc_pct)

    if row['emg_mean'] > emg_thr and row['acc_mean'] > acc_thr:
        return 'movement'        # hoge EMG én beweging = waarschijnlijk bewegingsartefact
    elif row['emg_mean'] > emg_thr:
        return 'emg_dominant'    # alleen hoge EMG = spierartefact
    elif row['acc_mean'] > acc_thr:
        return 'acc_dominant'    # alleen hoge beweging = motorisch artefact
    else:
        return 'clean'           # geen verhoogde EMG of beweging = schoon EEG-event


# ══════════════════════════════════════════════════════════════════════════════
# Fase 8 — Pipeline voor één nacht
# ══════════════════════════════════════════════════════════════════════════════

def process_one_night(edf_path, subject_id, night_id):
    """
    Verwerkt één EDF bestand van begin tot eind.

    Stappen:
    1. Laad en filter signalen
    2. Bereken CWT power matrix (voor beide EEG kanalen)
    3. Extraheer ridge
    4. Bereken freq_shift t.o.v. lokale baseline
    5. Detecteer kandidaat-events
    6. Bereken features per event
    7. Label artefacten
    8. Geef terug als DataFrame
    """
    # check of dit bestand al verwerkt is
    night_out = output_dir / f"{subject_id}_{night_id}_events.csv"
    if night_out.exists():
        print(f"  Al verwerkt, overgeslagen: {night_out.name}")
        return pd.read_csv(night_out)   # laad bestaand resultaat
    
    print(f"\n{'='*60}")
    print(f"Verwerken: {subject_id} / {night_id}")
    print(f"Bestand:   {edf_path}")

    # ── stap 1: laden en preprocessen ──
    print("  [1/6] Laden en filteren...")
    raw     = load_night(edf_path)
    signals = preprocess_signals(raw)
    n_samp  = len(signals[EEG_CH[0]])  # totaal aantal samples in de nacht

    # ── stap 2+3: CWT berekenen + ridge extraheren (streaming) ──
    # we gebruiken het gemiddelde van links en rechts EEG
    # dit vermindert ruis en artefacten die maar één kant raken
    print(f"  [2/6] CWT berekenen ({n_samp/SFREQ/3600:.1f} uur data)...")
    eeg_avg = (signals[EEG_CH[0]] + signals[EEG_CH[1]]) / 2.0

    # streaming CWT: berekent ridge + bandpowers zonder volledige power matrix
    # geheugen: ~168 MB i.p.v. ~1.93 GiB
    ridge_freq, ridge_power, band_mean = compute_morlet_tf_streaming(
        eeg_avg, SFREQ, FREQS, BANDS
    )
    print("  [3/6] Ridge extraheren... (gedaan tijdens CWT)")

    # ── stap 4: freq_shift berekenen ──
    print("  [4/6] Lokale baseline berekenen...")
    freq_shift, baseline = compute_freq_shift(ridge_freq, SFREQ, ROLLING_SEC)

    # ── stap 5: events detecteren ──
    print("  [5/6] Kandidaat-events detecteren...")
    events = detect_events(ridge_freq, ridge_power, freq_shift, SFREQ)
    print(f"         → {len(events)} kandidaat-events gevonden")

    if len(events) == 0:
        print("  Geen events gevonden, sla deze nacht over.")
        return pd.DataFrame()

    # ── stap 6: features per event + artifact labeling ──
    print("  [6/6] Features berekenen per event...")

    # combineer EMG kanalen: gemiddelde van links en rechts
    emg_avg = (np.abs(signals[EMG_CH[0]]) + np.abs(signals[EMG_CH[1]])) / 2.0

    # bewegingsmagnitude: √(dX² + dY² + dZ²)
    acc = np.sqrt(
        signals[MOV_CH[0]]**2 +
        signals[MOV_CH[1]]**2 +
        signals[MOV_CH[2]]**2
    )

    # bereken features voor elk event
    records = []
    for start, end in events:
        feat = extract_event_features(
            start, end, band_mean,
            ridge_freq, ridge_power, freq_shift,
            emg_avg, acc, SFREQ
        )
        # voeg metadata toe
        feat['subject_id'] = subject_id
        feat['night_id']   = night_id
        records.append(feat)

    # maak DataFrame van alle events
    df = pd.DataFrame(records)

    # artifact labeling op basis van verdeling binnen deze nacht
    df['artifact_label'] = df.apply(
        lambda row: label_artifact(
            row,
            emg_vals=df['emg_mean'].values,
            acc_vals=df['acc_mean'].values
        ),
        axis=1   # axis=1 = pas toe per rij
    )

    print(f"  → {len(df)} events, "
          f"{(df['artifact_label']=='clean').sum()} schoon, "
          f"{(df['artifact_label']!='clean').sum()} artefact")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fase 9 — Batch verwerking van alle nachten
# ══════════════════════════════════════════════════════════════════════════════

def run_all_nights(base_dir):
    """
    Zoekt alle EDF bestanden op in base_dir, verwerkt ze één voor één,
    en slaat resultaten op per nacht en als één grote master tabel.
    """
    # zoek alleen naar bestanden die eindigen op _psg.edf
    # dit sluit BATT.edf en andere niet-PSG bestanden uit
    # filter op T0_ zodat alleen baseline nachten worden verwerkt
    all_edf = sorted(f for f in base_dir.rglob("*_psg.edf") if "_T0_" in f.name)
    
    print(f"Gevonden: {len(all_edf)} PSG bestanden")
    
    if len(all_edf) == 0:
        print("Geen PSG bestanden gevonden. Controleer base_dir.")
        return pd.DataFrame()

    # begrens tot MAX_PARTICIPANTS voor pilot
    all_edf = all_edf[:MAX_PARTICIPANTS]
    print(f"Verwerken: {len(all_edf)} bestanden (MAX_PARTICIPANTS={MAX_PARTICIPANTS})")

    all_dfs = []

    for i, edf_path in enumerate(all_edf):
        # haal subject_id en night_id uit de bestandsnaam
        # bestandsnaam: bnbd_nsr_03554_T0_N1_psg.edf
        # stem (zonder .edf): bnbd_nsr_03554_T0_N1_psg
        stem = edf_path.stem                      
        parts = stem.replace("_psg", "").split("_")
        
        subject_id = f"bnbd_nsr_{parts[2]}"         
        night_id   = f"{parts[3]}_{parts[4]}"       

        try:
            df = process_one_night(edf_path, subject_id, night_id)

            if df.empty:
                continue

            night_out = output_dir / f"{subject_id}_{night_id}_events.csv"
            df.to_csv(night_out, index=False)
            print(f"  Opgeslagen: {night_out.name}")

            all_dfs.append(df)

        except Exception as e:
            print(f"  FOUT bij {edf_path.name}: {e}")
            continue

    if not all_dfs:
        print("Geen events gevonden over alle nachten.")
        return pd.DataFrame()

    master = pd.concat(all_dfs, ignore_index=True)
    master_out = output_dir / "master_events.csv"
    master.to_csv(master_out, index=False)

    print(f"\n{'='*60}")
    print(f"KLAAR")
    print(f"Totaal events:     {len(master)}")
    print(f"Deelnemers:        {master['subject_id'].nunique()}")
    print(f"Schone events:     {(master['artifact_label']=='clean').sum()}")
    print(f"Artefact events:   {(master['artifact_label']!='clean').sum()}")
    print(f"Master tabel:      {master_out}")

    return master

# ══════════════════════════════════════════════════════════════════════════════
# Kwaliteitscontrole — controleer of de pipeline werkt
# ══════════════════════════════════════════════════════════════════════════════

def qc_summary(master):
    """
    Print een overzicht per nacht zodat je kunt controleren of de
    detector realistisch werkt.

    Normen voor microarousals:
    - 5–30 events per uur slaap is realistisch
    - < 2 events/uur = detector te streng of te weinig data
    - > 100 events/uur = detector te gevoelig of veel ruis
    """
    print("\nKwaliteitscontrole per nacht:")
    print("-" * 60)

    for (subj, night), g in master.groupby(['subject_id', 'night_id']):
        n_events  = len(g)
        n_clean   = (g['artifact_label'] == 'clean').sum()
        mean_dur  = g['duration'].mean()
        mean_freq = g['mean_ridge_freq'].mean()

        # schat events per uur (aanname: alles is slaap)
        # verbeteren zodra je slaapstadia hebt
        total_dur_hr = g['duration'].sum() / 3600
        # gebruik signaallengte als proxy voor slaaptijd
        events_per_hr = n_events / max(total_dur_hr * 10, 1)

        print(f"{subj} / {night}:")
        print(f"  events totaal:    {n_events}")
        print(f"  schoon:           {n_clean} ({100*n_clean/n_events:.0f}%)")
        print(f"  gem. duur:        {mean_dur:.1f} sec")
        print(f"  gem. ridge freq:  {mean_freq:.1f} Hz")

    print("-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# START — voer de pipeline uit
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import winsound, time

    all_masters = []

    for idx, base_dir in enumerate(BASE_DIRS):
        group_name = base_dir.name
        print(f"\n{'#'*60}")
        print(f"# GROEP {idx+1}/3: {group_name}")
        print(f"{'#'*60}")

        master = run_all_nights(base_dir)

        if not master.empty:
            master['group'] = group_name
            all_masters.append(master)
            qc_summary(master)

        # notificatie: beep + bericht
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        if idx < len(BASE_DIRS) - 1:
            next_group = BASE_DIRS[idx + 1].name
            print(f"\n>>> KLAAR MET {group_name} — nu start {next_group} <<<\n")
        else:
            print(f"\n>>> ALLE GROEPEN KLAAR <<<\n")

    if all_masters:
        combined = pd.concat(all_masters, ignore_index=True)
        combined_out = output_dir / "master_events_ALL_GROUPS.csv"
        combined.to_csv(combined_out, index=False)
        print(f"Gecombineerde master tabel: {combined_out}")
        print(f"Totaal events alle groepen: {len(combined)}")
