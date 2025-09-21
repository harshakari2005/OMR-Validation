import re, pandas as pd

def clean_cell(s):
    if pd.isna(s):
        return None
    t = str(s).strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1].strip()
    return t

def parse_cell(cell):
    if cell is None:
        return None
    t = clean_cell(cell)
    if t is None:
        return None
    m = re.search(r"(\d{1,3})", t)
    if not m:
        return None
    q = int(m.group(1))
    idx = m.end()
    tail = t[idx:]
    tail_letters = re.findall(r"([A-Za-z])", tail)
    if tail_letters:
        ans = ''.join([c.upper() for c in tail_letters])
    else:
        letters = re.findall(r"([A-Za-z])", t)
        ans = ''.join([c.upper() for c in letters if not c.isdigit()])
    return q, ans

def parse_excel_to_json(sheets):
    converted = {}
    for sheet_name, df in sheets.items():
        mapping = {}
        for col in df.columns:
            for val in df[col].tolist():
                parsed = parse_cell(val)
                if parsed is None:
                    continue
                q, a = parsed
                mapping[str(q)] = a
        converted[sheet_name] = {"answers": mapping, "version_name": sheet_name}
    return converted
