"""Data cleaning and normalization for JoSAA cutoff records.

Filters raw tables according to rules.md:
- Drops P-suffix ranks (preparatory seats)
- Keeps only non-PwD seat types (OPEN, EWS, OBC-NCL, SC, ST)
- Keeps only standard quotas (AI, HS, OS)
- Normalizes institute and academic program names
- Assigns institute_type (IIT, NIT, IIIT, GFTI) and state
- Exports data/processed/institutes.csv and cleaned cutoffs table
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Map varying institute names across years to clean canonical names
INSTITUTE_RENAME_MAP: Dict[str, str] = {
    "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY SENAPATI MANIPUR": "Indian Institute of Information Technology Manipur",
    "Indian Institute of Information Technology(IIIT) Kilohrad, Sonepat, Haryana": "Indian Institute of Information Technology Sonepat",
    "Indian Institute of Information Technology, Sonepat": "Indian Institute of Information Technology Sonepat",
    "Indian Institute of Information Technology(IIIT) Kalyani, West Bengal": "Indian Institute of Information Technology Kalyani",
    "Indian Institute of Information Technology, Kalyani": "Indian Institute of Information Technology Kalyani",
    "Indian Institute of Information Technology(IIIT) Dharwad": "Indian Institute of Information Technology Dharwad",
    "Indian Institute of Information Technology, Dharwad": "Indian Institute of Information Technology Dharwad",
    "Indian Institute of Information Technology(IIIT) Kottayam": "Indian Institute of Information Technology Kottayam",
    "Indian Institute of Information Technology, Kottayam": "Indian Institute of Information Technology Kottayam",
    "Indian Institute of Information Technology(IIIT) Una, Himachal Pradesh": "Indian Institute of Information Technology Una",
    "Indian Institute of Information Technology, Una": "Indian Institute of Information Technology Una",
    "Indian Institute of Information Technology(IIIT), Vadodara, Gujrat": "Indian Institute of Information Technology Vadodara",
    "Indian Institute of Information Technology, Vadodara": "Indian Institute of Information Technology Vadodara",
    "Indian Institute of Information Technology (IIIT) Nagpur": "Indian Institute of Information Technology Nagpur",
    "Indian Institute of Information Technology (IIIT) Pune": "Indian Institute of Information Technology Pune",
    "Indian Institute of Information Technology (IIIT) Ranchi": "Indian Institute of Information Technology Ranchi",
    "Indian Institute of Information Technology (IIIT), Sri City, Chittoor": "Indian Institute of Information Technology Sri City",
    "Indian Institute of Information Technology, Sri City": "Indian Institute of Information Technology Sri City",
    "Indian Institute of Information Technology (IIIT)Kota, Rajasthan": "Indian Institute of Information Technology Kota",
    "Indian Institute of Information Technology, Kota": "Indian Institute of Information Technology Kota",
    "Indian institute of information technology, Raichur, Karnataka": "Indian Institute of Information Technology Raichur",
    "Indian Institute of Information Technology Design & Manufacturing Kurnool, Andhra Pradesh": "Indian Institute of Information Technology, Design & Manufacturing Kurnool",
    "Indian Institute of Information Technology, Design & Manufacturing, Kancheepuram": "Indian Institute of Information Technology, Design & Manufacturing Kancheepuram",
    "Pt. Dwarka Prasad Mishra Indian Institute of Information Technology, Design & Manufacture Jabalpur": "IIITDM Jabalpur",
    "Atal Bihari Vajpayee Indian Institute of Information Technology & Management Gwalior": "ABV-IIITM Gwalior",
    "Central University of Rajasthan, Rajasthan": "Central University of Rajasthan",
    "CU Jharkhand": "Central University of Jharkhand",
    "Central institute of Technology Kokrajar, Assam": "Central Institute of Technology Kokrajhar",
    "Central Institute of Technology Kokrajar, Assam": "Central Institute of Technology Kokrajhar",
    "Gati Shakti Vishwavidyalaya, Vadodara": "Gati Shakti Vishwavidyalaya, Vadodara",
    "Institute of Technology, Guru Ghasidas Vishwavidyalaya (A Central University), Bilaspur, (C.G.)": "Guru Ghasidas Vishwavidyalaya, Bilaspur",
    "School of Studies of Engineering and Technology, Guru Ghasidas Vishwavidyalaya, Bilaspur": "Guru Ghasidas Vishwavidyalaya, Bilaspur",
    "Institute of Engineering and Technology, Dr. H. S. Gour University. Sagar (A Central University)": "Dr. Harisingh Gour University, Sagar",
    "National Institute of Foundry & Forge Technology, Hatia, Ranchi": "National Institute of Advanced Manufacturing Technology, Ranchi",
    "National Institute of Food Technology, Entrepreneurship and Management (NIFTEM) - Thanjavur": "NIFTEM Thanjavur",
    "National Institute of Food Technology Entrepreneurship and Management, Thanjavur": "NIFTEM Thanjavur",
    "lndian Institute of Food Processing Technology, Thanjavur, Tamil Naidu.": "NIFTEM Thanjavur",
    "National Institute of Food Technology Entrepreneurship and Management, Kundli": "NIFTEM Kundli",
    "National Institute of Food Technology Entrepreneurship and Management, Sonepat, Haryana": "NIFTEM Kundli",
    "Pondicherry Engineering College, Puducherry": "Puducherry Technological University, Puducherry",
    "HNB Garhwal University Srinagar (Garhwal)": "HNB Garhwal University, Srinagar",
    "School of Planning & Architecture: Vijayawada": "School of Planning & Architecture, Vijayawada",
}

# Explicit institute metadata mapping: (institute_type, state)
INSTITUTE_METADATA: Dict[str, Tuple[str, str]] = {
    # --- IITs (23) ---
    "Indian Institute of Technology Bhubaneswar": ("IIT", "Odisha"),
    "Indian Institute of Technology Bombay": ("IIT", "Maharashtra"),
    "Indian Institute of Technology Mandi": ("IIT", "Himachal Pradesh"),
    "Indian Institute of Technology Delhi": ("IIT", "Delhi"),
    "Indian Institute of Technology Indore": ("IIT", "Madhya Pradesh"),
    "Indian Institute of Technology Kharagpur": ("IIT", "West Bengal"),
    "Indian Institute of Technology Hyderabad": ("IIT", "Telangana"),
    "Indian Institute of Technology Jodhpur": ("IIT", "Rajasthan"),
    "Indian Institute of Technology Kanpur": ("IIT", "Uttar Pradesh"),
    "Indian Institute of Technology Madras": ("IIT", "Tamil Nadu"),
    "Indian Institute of Technology Gandhinagar": ("IIT", "Gujarat"),
    "Indian Institute of Technology Patna": ("IIT", "Bihar"),
    "Indian Institute of Technology Roorkee": ("IIT", "Uttarakhand"),
    "Indian Institute of Technology (ISM) Dhanbad": ("IIT", "Jharkhand"),
    "Indian Institute of Technology Ropar": ("IIT", "Punjab"),
    "Indian Institute of Technology (BHU) Varanasi": ("IIT", "Uttar Pradesh"),
    "Indian Institute of Technology Guwahati": ("IIT", "Assam"),
    "Indian Institute of Technology Bhilai": ("IIT", "Chhattisgarh"),
    "Indian Institute of Technology Goa": ("IIT", "Goa"),
    "Indian Institute of Technology Palakkad": ("IIT", "Kerala"),
    "Indian Institute of Technology Tirupati": ("IIT", "Andhra Pradesh"),
    "Indian Institute of Technology Jammu": ("IIT", "Jammu and Kashmir"),
    "Indian Institute of Technology Dharwad": ("IIT", "Karnataka"),

    # --- NITs (31 + IIEST Shibpur) ---
    "Dr. B R Ambedkar National Institute of Technology, Jalandhar": ("NIT", "Punjab"),
    "Malaviya National Institute of Technology Jaipur": ("NIT", "Rajasthan"),
    "Maulana Azad National Institute of Technology Bhopal": ("NIT", "Madhya Pradesh"),
    "Motilal Nehru National Institute of Technology Allahabad": ("NIT", "Uttar Pradesh"),
    "National Institute of Technology Agartala": ("NIT", "Tripura"),
    "National Institute of Technology Calicut": ("NIT", "Kerala"),
    "National Institute of Technology Delhi": ("NIT", "Delhi"),
    "National Institute of Technology Durgapur": ("NIT", "West Bengal"),
    "National Institute of Technology Goa": ("NIT", "Goa"),
    "National Institute of Technology Hamirpur": ("NIT", "Himachal Pradesh"),
    "National Institute of Technology Karnataka, Surathkal": ("NIT", "Karnataka"),
    "National Institute of Technology Meghalaya": ("NIT", "Meghalaya"),
    "National Institute of Technology Nagaland": ("NIT", "Nagaland"),
    "National Institute of Technology Patna": ("NIT", "Bihar"),
    "National Institute of Technology Puducherry": ("NIT", "Puducherry"),
    "National Institute of Technology Raipur": ("NIT", "Chhattisgarh"),
    "National Institute of Technology Sikkim": ("NIT", "Sikkim"),
    "National Institute of Technology Arunachal Pradesh": ("NIT", "Arunachal Pradesh"),
    "National Institute of Technology, Jamshedpur": ("NIT", "Jharkhand"),
    "National Institute of Technology, Kurukshetra": ("NIT", "Haryana"),
    "National Institute of Technology, Manipur": ("NIT", "Manipur"),
    "National Institute of Technology, Mizoram": ("NIT", "Mizoram"),
    "National Institute of Technology, Rourkela": ("NIT", "Odisha"),
    "National Institute of Technology, Silchar": ("NIT", "Assam"),
    "National Institute of Technology, Srinagar": ("NIT", "Jammu and Kashmir"),
    "National Institute of Technology, Tiruchirappalli": ("NIT", "Tamil Nadu"),
    "National Institute of Technology, Uttarakhand": ("NIT", "Uttarakhand"),
    "National Institute of Technology, Warangal": ("NIT", "Telangana"),
    "Sardar Vallabhbhai National Institute of Technology, Surat": ("NIT", "Gujarat"),
    "Visvesvaraya National Institute of Technology, Nagpur": ("NIT", "Maharashtra"),
    "National Institute of Technology, Andhra Pradesh": ("NIT", "Andhra Pradesh"),
    "Indian Institute of Engineering Science and Technology, Shibpur": ("NIT", "West Bengal"),

    # --- IIITs ---
    "ABV-IIITM Gwalior": ("IIIT", "Madhya Pradesh"),
    "IIITDM Jabalpur": ("IIIT", "Madhya Pradesh"),
    "Indian Institute of Information Technology, Design & Manufacturing Kancheepuram": ("IIIT", "Tamil Nadu"),
    "Indian Institute of Information Technology, Design & Manufacturing Kurnool": ("IIIT", "Andhra Pradesh"),
    "Indian Institute of Information Technology, Allahabad": ("IIIT", "Uttar Pradesh"),
    "Indian Institute of Information Technology Lucknow": ("IIIT", "Uttar Pradesh"),
    "Indian Institute of Information Technology Bhagalpur": ("IIIT", "Bihar"),
    "Indian Institute of Information Technology Bhopal": ("IIIT", "Madhya Pradesh"),
    "Indian Institute of Information Technology Dharwad": ("IIIT", "Karnataka"),
    "Indian Institute of Information Technology Guwahati": ("IIIT", "Assam"),
    "Indian Institute of Information Technology Kalyani": ("IIIT", "West Bengal"),
    "Indian Institute of Information Technology Kota": ("IIIT", "Rajasthan"),
    "Indian Institute of Information Technology Kottayam": ("IIIT", "Kerala"),
    "Indian Institute of Information Technology Manipur": ("IIIT", "Manipur"),
    "Indian Institute of Information Technology Nagpur": ("IIIT", "Maharashtra"),
    "Indian Institute of Information Technology Pune": ("IIIT", "Maharashtra"),
    "Indian Institute of Information Technology Raichur": ("IIIT", "Karnataka"),
    "Indian Institute of Information Technology Ranchi": ("IIIT", "Jharkhand"),
    "Indian Institute of Information Technology Sonepat": ("IIIT", "Haryana"),
    "Indian Institute of Information Technology Sri City": ("IIIT", "Andhra Pradesh"),
    "Indian Institute of Information Technology Surat": ("IIIT", "Gujarat"),
    "Indian Institute of Information Technology Tiruchirappalli": ("IIIT", "Tamil Nadu"),
    "Indian Institute of Information Technology Una": ("IIIT", "Himachal Pradesh"),
    "Indian Institute of Information Technology Vadodara": ("IIIT", "Gujarat"),
    "Indian Institute of Information Technology, Agartala": ("IIIT", "Tripura"),
    "Indian Institute of Information Technology, Vadodara International Campus Diu (IIITVICD)": ("IIIT", "Dadra and Nagar Haveli and Daman and Diu"),
    "International Institute of Information Technology, Bhubaneswar": ("IIIT", "Odisha"),
    "International Institute of Information Technology, Naya Raipur": ("IIIT", "Chhattisgarh"),

    # --- GFTIs ---
    "Assam University, Silchar": ("GFTI", "Assam"),
    "Birla Institute of Technology, Deoghar Off-Campus": ("GFTI", "Jharkhand"),
    "Birla Institute of Technology, Mesra, Ranchi": ("GFTI", "Jharkhand"),
    "Birla Institute of Technology, Patna Off-Campus": ("GFTI", "Bihar"),
    "Central University of Haryana": ("GFTI", "Haryana"),
    "Central University of Jammu": ("GFTI", "Jammu and Kashmir"),
    "Central University of Jharkhand": ("GFTI", "Jharkhand"),
    "Central University of Rajasthan": ("GFTI", "Rajasthan"),
    "Central Institute of Technology Kokrajhar": ("GFTI", "Assam"),
    "Chhattisgarh Swami Vivekanada Technical University, Bhilai (CSVTU Bhilai)": ("GFTI", "Chhattisgarh"),
    "Dr. Harisingh Gour University, Sagar": ("GFTI", "Madhya Pradesh"),
    "Gati Shakti Vishwavidyalaya, Vadodara": ("GFTI", "Gujarat"),
    "Ghani Khan Choudhary Institute of Engineering and Technology, Malda, West Bengal": ("GFTI", "West Bengal"),
    "Gurukula Kangri Vishwavidyalaya, Haridwar": ("GFTI", "Uttarakhand"),
    "Guru Ghasidas Vishwavidyalaya, Bilaspur": ("GFTI", "Chhattisgarh"),
    "HNB Garhwal University, Srinagar": ("GFTI", "Uttarakhand"),
    "Indian Institute of Carpet Technology, Bhadohi": ("GFTI", "Uttar Pradesh"),
    "Indian Institute of Handloom Technology, Salem": ("GFTI", "Tamil Nadu"),
    "Indian Institute of Handloom Technology(IIHT), Varanasi": ("GFTI", "Uttar Pradesh"),
    "Institute of Chemical Technology, Mumbai: Indian Oil Odisha Campus, Bhubaneswar": ("GFTI", "Odisha"),
    "Institute of Infrastructure, Technology, Research and Management-Ahmedabad": ("GFTI", "Gujarat"),
    "Islamic University of Science and Technology Kashmir": ("GFTI", "Jammu and Kashmir"),
    "J.K. Institute of Applied Physics & Technology, Department of Electronics & Communication, University of Allahabad- Allahabad": ("GFTI", "Uttar Pradesh"),
    "Jawaharlal Nehru University, Delhi": ("GFTI", "Delhi"),
    "Mizoram University, Aizawl": ("GFTI", "Mizoram"),
    "National Institute of Advanced Manufacturing Technology, Ranchi": ("GFTI", "Jharkhand"),
    "National Institute of Electronics and Information Technology, Ajmer (Rajasthan)": ("GFTI", "Rajasthan"),
    "National Institute of Electronics and Information Technology, Aurangabad (Maharashtra)": ("GFTI", "Maharashtra"),
    "National Institute of Electronics and Information Technology, Gorakhpur (UP)": ("GFTI", "Uttar Pradesh"),
    "National Institute of Electronics and Information Technology, Patna (Bihar)": ("GFTI", "Bihar"),
    "National Institute of Electronics and Information Technology, Ropar (Punjab)": ("GFTI", "Punjab"),
    "NIFTEM Kundli": ("GFTI", "Haryana"),
    "NIFTEM Thanjavur": ("GFTI", "Tamil Nadu"),
    "North Eastern Regional Institute of Science and Technology, Nirjuli-791109 (Itanagar),Arunachal Pradesh": ("GFTI", "Arunachal Pradesh"),
    "North-Eastern Hill University, Shillong": ("GFTI", "Meghalaya"),
    "Puducherry Technological University, Puducherry": ("GFTI", "Puducherry"),
    "Punjab Engineering College, Chandigarh": ("GFTI", "Chandigarh"),
    "Rajiv Gandhi National Aviation University, Fursatganj, Amethi (UP)": ("GFTI", "Uttar Pradesh"),
    "Sant Longowal Institute of Engineering and Technology": ("GFTI", "Punjab"),
    "School of Engineering, Tezpur University, Napaam, Tezpur": ("GFTI", "Assam"),
    "School of Planning & Architecture, Bhopal": ("GFTI", "Madhya Pradesh"),
    "School of Planning & Architecture, New Delhi": ("GFTI", "Delhi"),
    "School of Planning & Architecture, Vijayawada": ("GFTI", "Andhra Pradesh"),
    "Shri G. S. Institute of Technology and Science Indore": ("GFTI", "Madhya Pradesh"),
    "Shri Mata Vaishno Devi University, Katra, Jammu & Kashmir": ("GFTI", "Jammu and Kashmir"),
    "University of Hyderabad": ("GFTI", "Telangana"),
}

VALID_QUOTAS = {"AI", "HS", "OS"}
VALID_SEAT_TYPES = {"OPEN", "EWS", "OBC-NCL", "SC", "ST"}


def normalize_institute_name(name: str) -> str:
    """Normalize institute name using the rename map and fallback string cleaning."""
    clean_name = str(name).strip()
    return INSTITUTE_RENAME_MAP.get(clean_name, clean_name)


def infer_institute_metadata(institute: str) -> Tuple[str, str]:
    """Return (institute_type, state) for the given normalized institute name."""
    if institute in INSTITUTE_METADATA:
        return INSTITUTE_METADATA[institute]

    # Heuristic fallback if a new institute appears
    inst_lower = institute.lower()
    if "indian institute of technology" in inst_lower and "information" not in inst_lower:
        inst_type = "IIT"
    elif "national institute of technology" in inst_lower or "nit" in inst_lower:
        inst_type = "NIT"
    elif "information technology" in inst_lower or "iiit" in inst_lower:
        inst_type = "IIIT"
    else:
        inst_type = "GFTI"

    # Fallback state detection
    state = "Other"
    for s in [
        "Maharashtra", "Delhi", "Uttar Pradesh", "Tamil Nadu", "Karnataka",
        "Telangana", "Gujarat", "Rajasthan", "West Bengal", "Madhya Pradesh",
        "Punjab", "Haryana", "Bihar", "Odisha", "Kerala", "Assam", "Jharkhand",
        "Uttarakhand", "Himachal Pradesh", "Jammu and Kashmir", "Chhattisgarh",
        "Goa", "Tripura", "Meghalaya", "Manipur", "Mizoram", "Nagaland",
        "Sikkim", "Arunachal Pradesh", "Puducherry", "Chandigarh"
    ]:
        if s.lower() in inst_lower:
            state = s
            break

    return inst_type, state


def normalize_program_name(prog: str) -> str:
    """Normalize program names by removing degree duration and boilerplate strings."""
    if not isinstance(prog, str):
        return ""
    text = prog.strip()

    # Identify special degree qualifications before stripping parentheticals
    is_dual = "dual degree" in text.lower() or "bachelor and master" in text.lower()
    is_integ_mtech = "integrated master of technology" in text.lower()
    is_integ_msc = "integrated master of science" in text.lower()
    is_bs_ms = "bs-ms" in text.lower() or "bachelor of science and master of science" in text.lower()
    is_bs = "bachelor of science" in text.lower() and not is_bs_ms
    is_barch = "bachelor of architecture" in text.lower()
    is_bplan = "bachelor of planning" in text.lower()

    # Strip duration and degree parentheticals: e.g. "(4 Years, Bachelor of Technology)"
    text = re.sub(r"\s*\(\s*\d+\s*Years?[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Append canonical degree qualifier to keep separate JoSAA seat choices distinct
    if is_dual:
        text = f"{text} (Dual Degree)"
    elif is_integ_mtech:
        text = f"{text} (Integrated M.Tech)"
    elif is_integ_msc:
        text = f"{text} (Integrated M.Sc)"
    elif is_bs_ms:
        text = f"{text} (BS-MS Dual Degree)"
    elif is_bs:
        text = f"{text} (BS)"
    elif is_barch:
        text = f"{text} (B.Arch)"
    elif is_bplan:
        text = f"{text} (B.Plan)"

    return text



def parse_clean_rank(val: object) -> Optional[int]:
    """Parse rank value. Returns None if preparatory ('P' suffix) or unparseable."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Drop preparatory ranks ending in P or containing P
    if "P" in s.upper():
        return None
    # Strip non-digits
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    try:
        rank = int(digits)
        return rank if rank > 0 else None
    except ValueError:
        return None


def clean_single_year_file(file_path: Path) -> pd.DataFrame:
    """Clean a single raw CSV file."""
    # Match pattern <year>_round<round>.csv
    match = re.search(r"(\d{4})_round(\d+)", file_path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {file_path.name}")
    year = int(match.group(1))
    round_no = int(match.group(2))

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Institute", "Academic Program Name", "Closing Rank"])

    # Clean and filter ranks
    df["opening_rank"] = df["Opening Rank"].apply(parse_clean_rank)
    df["closing_rank"] = df["Closing Rank"].apply(parse_clean_rank)
    df = df.dropna(subset=["opening_rank", "closing_rank"])
    df["opening_rank"] = df["opening_rank"].astype(int)
    df["closing_rank"] = df["closing_rank"].astype(int)

    # Filter Quotas
    df["Quota"] = df["Quota"].astype(str).str.strip().str.upper()
    df = df[df["Quota"].isin(VALID_QUOTAS)].copy()

    # Filter Seat Types (exclude PwD categories)
    df["Seat Type"] = df["Seat Type"].astype(str).str.strip()
    df = df[df["Seat Type"].isin(VALID_SEAT_TYPES)].copy()

    # Normalize Gender
    df["Gender"] = df["Gender"].astype(str).str.strip()
    df["gender"] = df["Gender"].apply(
        lambda g: "Female-only" if "female" in g.lower() else "Gender-Neutral"
    )

    # Normalize Names
    df["institute"] = df["Institute"].apply(normalize_institute_name)
    df["program"] = df["Academic Program Name"].apply(normalize_program_name)

    # Exclude B.Arch, B.Planning, and B.Des (they do not use JEE Main Paper 1 CRL)
    is_arch_or_plan = (
        df["program"].str.contains("Architecture|Planning", case=False, na=False)
    ) & (~df["program"].str.contains("Naval Architecture", case=False, na=False))
    is_bdes = (
        df["program"].str.strip().str.lower() == "bachelor of design"
    ) | df["program"].str.contains(r"\b(?:b\.?des|bachelor of design)\b", case=False, na=False)
    df = df[~(is_arch_or_plan | is_bdes)].copy()

    df["quota"] = df["Quota"]
    df["seat_type"] = df["Seat Type"]

    # Assign institute type and state
    meta = [infer_institute_metadata(inst) for inst in df["institute"]]
    df["institute_type"] = [m[0] for m in meta]
    df["state"] = [m[1] for m in meta]

    df["year"] = year
    df["round"] = round_no

    target_cols = [
        "year",
        "round",
        "institute",
        "institute_type",
        "state",
        "program",
        "quota",
        "seat_type",
        "gender",
        "opening_rank",
        "closing_rank",
    ]
    cleaned = df[target_cols].drop_duplicates(
        subset=["year", "institute", "program", "quota", "seat_type", "gender"]
    )
    return cleaned


def build_and_save_institutes_csv(df_all: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Build and save unique institutes table with type and state."""
    inst_df = (
        df_all[["institute", "institute_type", "state"]]
        .drop_duplicates()
        .sort_values(by=["institute_type", "institute"])
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inst_df.to_csv(output_path, index=False)
    return inst_df


def clean_all_raw_files(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Clean all raw files in raw_dir, output institutes.csv and return combined dataframe."""
    raw_files = sorted(raw_dir.glob("*_round*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")

    dfs = [clean_single_year_file(f) for f in raw_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Save institutes.csv
    institutes_csv = processed_dir / "institutes.csv"
    build_and_save_institutes_csv(combined, institutes_csv)
    print(f"Saved {institutes_csv} with {combined['institute'].nunique()} unique institutes.")

    return combined


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    cleaned = clean_all_raw_files(raw_dir, processed_dir)
    print(f"Cleaned {len(cleaned)} total rows across {cleaned['year'].nunique()} years.")


if __name__ == "__main__":
    main()

