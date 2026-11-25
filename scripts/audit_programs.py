import pandas as pd

df = pd.read_parquet("data/processed/cutoffs.parquet")
progs_by_year = {y: set(df[df["year"] == y]["program"]) for y in sorted(df["year"].unique())}
all_progs = set().union(*progs_by_year.values())
print(f"Total distinct normalized programs: {len(all_progs)}")
for y in sorted(progs_by_year.keys()):
    print(f"Year {y}: {len(progs_by_year[y])} programs")

# Programs in 2025 not in 2024
new_2025 = progs_by_year[2025] - progs_by_year[2024]
print(f"\nPrograms in 2025 but not in 2024: {len(new_2025)}")
for p in sorted(list(new_2025))[:15]:
    print(" +", p)

# Check seat counts
print("\nChecking for seat count columns in cutoffs.parquet:")
seat_cols = [c for c in df.columns if "seat" in c.lower() or "capacity" in c.lower() or "intake" in c.lower()]
print("Seat-related columns:", seat_cols)
