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

base_dir   = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
output_dir = Path(r"C:\Users\zafar\Documents\bnbd_output3")
output_dir.mkdir(exist_ok=True)   # maak output map aan als die nog niet bestaat

MAX_PARTICIPANTS = 5         # hoeveel deelnemers je wilt verwerken (5 voor pilot)

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

# EEG frequentiebanden — gebruikt voor feature extractie per event
BANDS = {
    'delta': (0.5,  4.0),   # diepe slaap
    'theta': (4.0,  8.0),   # lichte slaap / drowsiness
    'alpha': (8.0,  13.0),  # ontspanning / arousal
    'beta':  (13.0, 35.0),  # activatie / alertheid
}

ROLLING_SEC            = 60.0   # baseline venster: vergelijk met de vorige 60 seconden
AROUSAL_FREQ_THRESHOLD = 3.0    # ridge moet > 3 Hz boven baseline springen = activatie
AROUSAL_MIN_DUR        = 3.0    # event moet minimaal 3 seconden duren
AROUSAL_MAX_DUR        = 30.0   # event mag maximaal 30 seconden duren


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
# Fase 2 — Morlet CWT (identiek aan Scoring Hero van supervisor)
# ══════════════════════════════════════════════════════════════════════════════

def compute_morlet_tf(signal, srate, freqs, n_cycles=None, L2normalize=False):
    """
    Berekent de Continuous Wavelet Transform via Morlet wavelets.

    Voor elke frequentie: maak een golfje (wavelet) en kijk hoe goed
    dat golfje past op het signaal op elk tijdstip.
    Hoe beter de match → hogere power.

    Output: power matrix van shape (n_freqs, n_samples)
    Elke rij = één frequentie, elke kolom = één tijdstip
    Kleur in Scoring Hero = de waarde in deze matrix
    """
    freqs = np.asarray(freqs)

    # n_cycles bepaalt de breedte van het golfje:
    # - lage frequentie → breed golfje → goede tijdresolutie
    # - hoge frequentie → smal golfje → goede frequentieresolutie
    if n_cycles is None:
        n_cycles_arr = np.maximum(3.0, freqs / 2.0)   # slimme keuze van supervisor
    elif np.isscalar(n_cycles):
        n_cycles_arr = np.full(len(freqs), float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)

    n_samples = len(signal)

    # verwijder DC offset: vlakke lijn weghalen zodat het niet lekt
    # naar lage frequenties (zou nep-delta power geven)
    signal = signal - np.mean(signal)

    # zet signaal om naar frequentiedomein (FFT = snelle fourier transform)
    # dit is de basis voor de convolutie met elke wavelet
    signal_fft = np.fft.fft(signal)

    # frequentievector die hoort bij de FFT output
    fft_freqs = np.fft.fftfreq(n_samples, d=1.0 / srate)

    # lege matrix om power in op te slaan
    power = np.empty((len(freqs), n_samples), dtype=np.float32)

    # bereken voor elke doelfrequentie de power
    for i, freq in enumerate(tqdm(freqs, desc="CWT", leave=False)):
        # sigma_f = breedte van de Gaussiaanse envelop in frequentiedomein
        sigma_f = freq / n_cycles_arr[i]

        # maak de wavelet als Gaussiaanse bel rondom de doelfrequentie
        # hoge waarde bij freq, laag daaromheen
        wavelet_fft = np.exp(-0.5 * ((fft_freqs - freq) / sigma_f) ** 2)

        # optioneel: normaliseer zodat power vergelijkbaar is over frequenties
        if L2normalize:
            wavelet_fft /= np.sqrt(np.sum(wavelet_fft ** 2))

        # vermenigvuldig signaal met wavelet in frequentiedomein
        # = convolutie in tijdsdomein (sneller via FFT)
        analytic = np.fft.ifft(signal_fft * wavelet_fft)

        # power = gekwadrateerde magnitude van het analytische signaal
        # dit geeft de instantane energie op elk tijdstip
        power[i] = np.abs(analytic) ** 2

    return power   # shape: (n_freqs, n_samples)


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — Ridge extractie
# ══════════════════════════════════════════════════════════════════════════════

def extract_ridge(power, freqs):
    """
    Trekt de ridge uit de power matrix.

    Ridge = per tijdstip de frequentie met de hoogste power.
    Dit is de zwarte lijn die je ziet in Scoring Hero.

    power: (n_freqs, n_samples)
    freqs: (n_freqs,)

    Geeft terug:
    - ridge_freq:  (n_samples,) — dominante frequentie per tijdstip in Hz
    - ridge_power: (n_samples,) — power op die dominante frequentie
    """
    # argmax over axis=0 = per tijdstip (kolom) de rij met hoogste waarde
    ridge_idx = np.argmax(power, axis=0)   # geeft indices, niet Hz-waarden

    # zet index om naar Hz via de freqs vector
    ridge_freq = freqs[ridge_idx]          # shape: (n_samples,)

    # pak de power op die index per tijdstip
    # np.arange(n) maakt [0, 1, 2, ...] voor de kolomindex
    ridge_power = power[ridge_idx, np.arange(power.shape[1])]

    return ridge_freq, ridge_power


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

def extract_event_features(start, end, power, freqs, ridge_freq,
                            ridge_power, freq_shift, emg_signal, acc_signal,
                            srate):
    """
    Berekent alle features voor één event.

    Gebruikt de power matrix (al berekend) om bandpowers te halen.
    Dit is efficiënt: CWT wordt slechts één keer berekend per nacht.

    Features:
    - timing: start, einde, duur
    - ridge: gemiddelde freq, freq_shift, peak power
    - bandpowers: delta, theta, alpha, beta (genormaliseerd)
    - ratio: fast/slow — hoe actief was het EEG?
    - EMG: gemiddelde spieractiviteit tijdens event
    - beweging: gemiddelde accelerometer magnitude
    """
    # slice de power matrix op het tijdvenster van dit event
    # power[:, start:end] = alle frequenties, alleen de samples van dit event
    pmat = power[:, start:end]   # shape: (n_freqs, event_duur)

    # ── bandpowers ──
    def bandpower(lo, hi):
        # selecteer rijen (frequenties) die in de band vallen
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            return 0.0
        # gemiddelde power over die frequenties en over de tijd
        return float(pmat[mask, :].mean())

    delta = bandpower(*BANDS['delta'])   # 0.5–4 Hz
    theta = bandpower(*BANDS['theta'])   # 4–8 Hz
    alpha = bandpower(*BANDS['alpha'])   # 8–13 Hz
    beta  = bandpower(*BANDS['beta'])    # 13–35 Hz
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

    # ── stap 2: CWT berekenen ──
    # we gebruiken het gemiddelde van links en rechts EEG
    # dit vermindert ruis en artefacten die maar één kant raken
    print(f"  [2/6] CWT berekenen ({n_samp/SFREQ/3600:.1f} uur data)...")
    eeg_avg = (signals[EEG_CH[0]] + signals[EEG_CH[1]]) / 2.0

    # bereken de volledige power matrix: (n_freqs × n_samples)
    # dit is de zwaarste berekening — kan 1–5 minuten duren per nacht
    power = compute_morlet_tf(eeg_avg, SFREQ, FREQS)

    # ── stap 3: ridge extraheren ──
    print("  [3/6] Ridge extraheren...")
    ridge_freq, ridge_power = extract_ridge(power, FREQS)

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
            start, end, power, FREQS,
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

def run_all_nights():

    """
    Zoekt alle EDF bestanden op, verwerkt ze één voor één,
    en slaat resultaten op per nacht en als één grote master tabel.

    Structuur van de data map:
    base_dir/
    └── sub_001/
        └── nacht_1.edf
        └── nacht_2.edf
    └── sub_002/
        └── ...
    """
    # zoek alleen naar bestanden die eindigen op _psg.edf
    # dit sluit BATT.edf en andere niet-PSG bestanden uit
    all_edf = sorted(base_dir.rglob("*_psg.edf"))
    
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

    # voer de volledige pipeline uit
    master = run_all_nights()

    # toon kwaliteitscontrole als er events zijn
    if not master.empty:
        qc_summary(master)
        print("\nKolommen in master tabel:")
        print(master.columns.tolist())
        print("\nEerste 3 events:")
        print(master.head(3).to_string())

    # ── volgende stap: clustering ──
    # master is nu klaar voor UMAP + HDBSCAN
    # zie microarousal_pipeline.py voor cluster_events() en plot_umap()
