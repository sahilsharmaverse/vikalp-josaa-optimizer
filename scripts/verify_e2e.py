import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf
import pandas as pd
from src.eligibility import StudentProfile
from src.optimize import build_list
from src.pdf_report import build_pdf_report

print("--- END-TO-END VERIFICATION ---")

# Profile 1: Low-Rank Profile (Rank 45,000, CSE, NIT/IIIT, list_size=20)
p1 = StudentProfile(
    main_rank=45000,
    category="OPEN",
    gender="Gender-Neutral",
    home_state="Delhi",
    preferred_branches=["CSE"],
    institute_types=["NIT", "IIIT"],
    list_size=20
)
df1 = build_list(p1)
safe1 = int((df1["label"] == "Safe").sum())
dream1 = int((df1["label"] == "Dream").sum())
target1 = int((df1["label"] == "Target").sum())
ui_warning1 = (safe1 < 5)

pdf1_bytes = build_pdf_report(df1, p1, seat_chance=0.72)
reader1 = pypdf.PdfReader(io.BytesIO(pdf1_bytes))
text1 = " ".join([p.extract_text() for p in reader1.pages])
pdf_warning1 = "few or no safe options" in text1

print(f"Profile 1 (Low-Rank: CRL 45,000, CSE):")
print(f"  Choices: {len(df1)} (Dream: {dream1}, Target: {target1}, Safe: {safe1})")
print(f"  UI Warning Triggered: {ui_warning1}")
print(f"  PDF Warning Triggered: {pdf_warning1}")
print(f"  All P >= 0.05: {(df1['probability'] >= 0.05).all()}")
assert ui_warning1 is True, "UI warning should trigger for safe < 5"
assert pdf_warning1 is True, "PDF warning should trigger for safe < 5"
assert (df1["probability"] >= 0.05).all(), "All choices must meet 5% floor"

# Profile 2: High-Rank Profile (Rank 25,000, list_size=30)
p2 = StudentProfile(
    main_rank=25000,
    category="OPEN",
    gender="Gender-Neutral",
    home_state="Delhi",
    list_size=30
)
df2 = build_list(p2)
safe2 = int((df2["label"] == "Safe").sum())
dream2 = int((df2["label"] == "Dream").sum())
target2 = int((df2["label"] == "Target").sum())
ui_warning2 = (safe2 < 5)

pdf2_bytes = build_pdf_report(df2, p2, seat_chance=0.99)
reader2 = pypdf.PdfReader(io.BytesIO(pdf2_bytes))
text2 = " ".join([p.extract_text() for p in reader2.pages])
pdf_warning2 = "few or no safe options" in text2

print(f"\nProfile 2 (High-Rank: CRL 25,000, general):")
print(f"  Choices: {len(df2)} (Dream: {dream2} [{dream2/len(df2):.1%}], Target: {target2} [{target2/len(df2):.1%}], Safe: {safe2} [{safe2/len(df2):.1%}])")
print(f"  UI Warning Triggered: {ui_warning2}")
print(f"  PDF Warning Triggered: {pdf_warning2}")
print(f"  All P >= 0.05: {(df2['probability'] >= 0.05).all()}")
assert ui_warning2 is False, "UI warning should not trigger when safe >= 5"
assert pdf_warning2 is False, "PDF warning should not trigger when safe >= 5"
assert (df2["probability"] >= 0.05).all(), "All choices must meet 5% floor"
assert dream2 == 6, f"Expected 6 dream choices, got {dream2}"   # exactly 20% of 30
assert target2 == 15, f"Expected 15 target choices, got {target2}" # exactly 50% of 30
assert safe2 == 9, f"Expected 9 safe choices, got {safe2}"    # exactly 30% of 30

print("\nALL END-TO-END CHECKS PASSED!")
