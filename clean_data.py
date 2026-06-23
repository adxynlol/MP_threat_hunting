import pandas as pd
import numpy as np
import os

def report(label, before, after, notes):
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  Rows  : {before:,} → {after:,}  (removed {before - after:,})")
    for note in notes:
        print(f"  ✓ {note}")


# ─────────────────────────────────────────────
# 1. processed_ddos_data
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Cleaning: processed_ddos_data.csv")
print("="*55)

df = pd.read_csv("processed_ddos_data.csv", low_memory=False)
before = len(df)
notes = []

# Fwd Header Length and Fwd Header Length.1 contain negative sentinel values (-2).
# Header lengths are non-negative by definition — clip to 0.
for col in [" Fwd Header Length", "Fwd Header Length.1"]:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0)
        notes.append(f"Clipped {neg_count:,} negative values in '{col}' to 0")

df.to_csv("processed_ddos_data_cleaned.csv", index=False)
report("processed_ddos_data", before, len(df), notes)
print(f"  Saved → processed_ddos_data_cleaned.csv")


# ─────────────────────────────────────────────
# 2. unsw_suspicious_events
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Cleaning: unsw_suspicious_events.csv")
print("="*55)

df = pd.read_csv("unsw_suspicious_events.csv", low_memory=False)
before = len(df)
notes = []

# Drop duplicate rows — 98k duplicates make up ~60% of the dataset
dupes = df.duplicated().sum()
df = df.drop_duplicates()
notes.append(f"Dropped {dupes:,} duplicate rows")

# Drop constant columns (zero variance — carry no information)
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    notes.append(f"Dropped constant columns: {const_cols}")

df.to_csv("unsw_suspicious_events_cleaned.csv", index=False)
report("unsw_suspicious_events", before, len(df), notes)
print(f"  Saved → unsw_suspicious_events_cleaned.csv")


# ─────────────────────────────────────────────
# 3. Summary files — already clean, just copy
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Cleaning: unsw_train_summary + unsw_test_summary")
print("="*55)
for src, dst in [
    ("unsw_train_summary.csv",    "unsw_train_summary_cleaned.csv"),
    ("unsw_test_summary (1).csv", "unsw_test_summary_cleaned.csv"),
]:
    df = pd.read_csv(src, low_memory=False)
    df.to_csv(dst, index=False)
    print(f"  No issues found — saved as-is → {dst}")


# ─────────────────────────────────────────────
# Final quality check on all cleaned files
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("POST-CLEAN QUALITY CHECK")
print("="*55)
cleaned = [
    "processed_ddos_data_cleaned.csv",
    "unsw_suspicious_events_cleaned.csv",
    "unsw_train_summary_cleaned.csv",
    "unsw_test_summary_cleaned.csv",
]
print(f"\n  {'File':<42} {'Rows':>8} {'Dupes':>7} {'NaNs':>7} {'Infs':>7}")
print(f"  {'─'*42} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
for path in cleaned:
    df = pd.read_csv(path, low_memory=False)
    dupes = df.duplicated().sum()
    nans  = df.isnull().sum().sum()
    num   = df.select_dtypes(include=[np.number])
    infs  = int(np.isinf(num).sum().sum())
    print(f"  {path:<42} {len(df):>8,} {dupes:>7,} {nans:>7,} {infs:>7,}")
print()
