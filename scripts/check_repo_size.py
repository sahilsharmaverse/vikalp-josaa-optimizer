"""List top 20 largest tracked files and total repo size."""

import subprocess
from pathlib import Path

files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
file_sizes = []
total_size = 0

for f in files:
    p = Path(f)
    if p.exists():
        sz = p.stat().st_size
        file_sizes.append((f, sz))
        total_size += sz

file_sizes.sort(key=lambda x: x[1], reverse=True)

print(f"Total Tracked File Count: {len(file_sizes)}")
print(f"Total Tracked Repo Size : {total_size / (1024*1024):.2f} MB ({total_size:,} bytes)\n")
print(f"{'Rank':<5} | {'Size (KB)':<12} | {'File Path':<60}")
print("-" * 82)
for idx, (f, sz) in enumerate(file_sizes[:20], 1):
    flag = " [OVER 20MB!]" if sz > 20 * 1024 * 1024 else ""
    print(f"{idx:<5} | {sz / 1024:<12.2f} | {f:<60}{flag}")
