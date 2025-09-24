import re
import pandas as pd

INPUT_FILE = "resources/estrellas.txt"
FIXED_FILE = "resources/estrellas_cleaned.txt"
CLEANED_CSV = "resources/estrellas_cleaned.csv"

def insert_pipes_between_numbers(s: str) -> str:
    # Floats next to floats
    s = re.sub(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', r'\1|\2', s)
    # Ints next to floats or vice versa
    s = re.sub(r'(-?\d+)\s+(-?\d+\.\d+)', r'\1|\2', s)
    s = re.sub(r'(-?\d+\.\d+)\s+(-?\d+)', r'\1|\2', s)
    # Ints next to ints
    s = re.sub(r'(-?\d+)\s+(-?\d+)', r'\1|\2', s)
    return s

def insert_pipes_special_cases(s: str) -> str:
    # Between text token and ISO date
    s = re.sub(r'([A-Za-z_][A-Za-z0-9_ ]*?)\s+(\d{4}-\d{2}-\d{2})', r'\1|\2', s)
    # Between consecutive 'null' separated by 2+ spaces
    s = re.sub(r'(null)\s{2,}(null)', r'\1|\2', s, flags=re.IGNORECASE)
    return s

def safe_tokenize(line: str):
    line = line.strip().strip('|')
    line = insert_pipes_between_numbers(line)
    line = insert_pipes_special_cases(line)
    return [p for p in line.split('|')]

def fix_file(in_path: str, out_path: str):
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # Header
    header_raw = lines[0].strip().strip("|")
    header_cols = [c.strip() for c in header_raw.split("|")]
    n_cols = len(header_cols)

    fixed_lines = ["|".join(header_cols)]

    counter = 0
    total = len(lines) - 1

    for ln in lines[1:]:
        
        counter += 1
        print("Processing...", counter / total * 100 , end="\r")
        
        parts = safe_tokenize(ln)

        # If too few, split on 2+ spaces (but not single space, so "2MASS J..." stays intact)
        if len(parts) < n_cols:
            repaired = []
            for p in parts:
                if "  " in p:
                    subs = re.split(r"\s{2,}", p.strip())
                    repaired.extend(subs)
                else:
                    repaired.append(p.strip())
            parts = repaired

        # Normalize length
        if len(parts) < n_cols:
            parts += ["null"] * (n_cols - len(parts))
        elif len(parts) > n_cols:
            parts = parts[:n_cols]

        fixed_lines.append("|".join(p.strip() for p in parts))

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in fixed_lines:
            f.write(ln + "\n")

if __name__ == "__main__":
    # Step 1: clean file
    fix_file(INPUT_FILE, FIXED_FILE)