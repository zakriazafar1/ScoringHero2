from pathlib import Path

base_dir = Path(r"\\vs03.herseninstituut.knaw.nl\VS03-SandC-2\raw\bnbd\Data\eeg\NSR")
all_edf  = sorted(f for f in base_dir.rglob("*_psg.edf") if "_T0_" in f.stem)

print(f"Totaal T0 PSG bestanden: {len(all_edf)}")

participants = set(f.stem.split("_")[2] for f in all_edf)
print(f"Unieke participanten:    {len(participants)}")

for f in all_edf[:10]:
    print(f"  {f.stem}")