from pathlib import Path

base_root = Path("//vs03.herseninstituut.knaw.nl/VS03-SandC-2/raw/bnbd/Data/eeg")
cohorts   = ["SAV", "Prezens"]

participants_of_interest = {
    "18500", "20362", "20736", "21614", "21962", "23264", "23343"
}

found = []
for cohort in cohorts:
    cohort_dir = base_root / cohort
    if not cohort_dir.exists():
        print(f"Not accessible: {cohort_dir}")
        continue
    for f in sorted(cohort_dir.rglob("*_psg.edf")):
        if "_T0_" in f.stem:
            parts = f.stem.split("_")
            for p in parts:
                if p in participants_of_interest:
                    found.append((cohort, p, f))
                    break

print(f"Found {len(found)} T0 PSG EDF files across {len({pid for _, pid, _ in found})} participants:\n")
for cohort, pid, f in found:
    print(f"  [{cohort}] {pid}: {f.name}")

missing = participants_of_interest - {pid for _, pid, _ in found}
if missing:
    print(f"\nMissing ({len(missing)}): {sorted(missing)}")
else:
    print(f"\nAll {len(participants_of_interest)} participants found.")
