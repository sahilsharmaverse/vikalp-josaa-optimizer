"""Scan git history and working tree for secrets, credentials, personal info, and local paths."""

import os
from pathlib import Path
import re
import subprocess

ROOT_DIR = Path(".").resolve()

PATTERNS = {
    "local_path": re.compile(r"[C-Zc-z]:[\\/][Uu]sers[\\/][^\\/\s\"']+", re.IGNORECASE),
    "api_key": re.compile(r"(?:api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{8,}['\"]", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}"),
}

def scan_working_tree():
    print("=== SCANNING WORKING TREE ===")
    findings = []
    
    # Check for .env files
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip .git and .venv
        dirs[:] = [d for d in dirs if d not in [".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"]]
        for f in files:
            if ".env" in f.lower():
                findings.append(("ENV_FILE", os.path.join(root, f), 0, f"Found env file: {f}"))
            
            fpath = Path(root) / f
            # Check text files only
            if fpath.suffix.lower() in [".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".ini", ".toml", ".html", ".css", ".sh", ".bat"]:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as file:
                        for line_num, line in enumerate(file, 1):
                            for p_name, regex in PATTERNS.items():
                                matches = regex.findall(line)
                                if matches:
                                    # Filter benign emails like josaa.nic.in
                                    if p_name == "email":
                                        matches = [m for m in matches if not any(d in m.lower() for d in ["josaa.nic.in", "example.com", "streamlit.app", "github.com", "mit-license.org", "python.org"])]
                                    if p_name == "local_path":
                                        # Keep all matches
                                        pass
                                    if p_name == "phone":
                                        # Filter timestamps or large rank numbers like 20261127 or random floats
                                        matches = [m for m in matches if not (m.startswith("2026") or m.startswith("2025") or m.startswith("2024") or m.startswith("2023") or m.startswith("2022") or m.startswith("2021"))]
                                    if matches:
                                        rel_path = fpath.relative_to(ROOT_DIR)
                                        findings.append((p_name, str(rel_path), line_num, f"{line.strip()[:100]} (Matches: {matches})"))
                except Exception as e:
                    pass
    return findings

def scan_git_history():
    print("\n=== SCANNING GIT COMMIT HISTORY ===")
    history_findings = []
    
    # 1. Search git log diffs for local paths
    try:
        cmd = ["git", "log", "-S", "Users", "-p", "--oneline"]
        res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, errors="ignore")
        if res.stdout:
            for line in res.stdout.split("\n"):
                if any(k in line.lower() for k in ["users\\hp", "users/hp"]):
                    history_findings.append(("GIT_HISTORY_LOCAL_PATH", line[:120]))
    except Exception as e:
        history_findings.append(("ERROR", str(e)))

    # 2. Search git log for api_key, secret, token, password
    for kw in ["api_key", "secret", "password", "token", ".env"]:
        try:
            cmd = ["git", "log", "-i", "-S", kw, "--oneline"]
            res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, errors="ignore")
            lines = [l.strip() for l in res.stdout.split("\n") if l.strip()]
            if lines:
                # Inspect commits
                for l in lines[:5]:
                    history_findings.append((f"GIT_LOG_KEYWORD_{kw.upper()}", l))
        except Exception:
            pass
            
    return history_findings

if __name__ == "__main__":
    wt_findings = scan_working_tree()
    print(f"Working tree findings: {len(wt_findings)}")
    for kind, path, line, detail in wt_findings:
        print(f"[{kind}] {path}:{line} -> {detail}")
        
    git_findings = scan_git_history()
    print(f"\nGit history findings: {len(git_findings)}")
    for kind, detail in git_findings[:25]:
        print(f"[{kind}] {detail}")
