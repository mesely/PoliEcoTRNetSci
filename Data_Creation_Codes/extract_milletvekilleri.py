import pdfplumber
import json
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PDF_DIR = DATA_DIR / "meclis_donem_dagilim_pdf"
OUTPUT_JSON = DATA_DIR / "milletvekilleri.json"
OUTPUT_CSV = DATA_DIR / "milletvekilleri.csv"

def clean(text):
    if text is None:
        return ""
    text = re.sub(r'\[\d+\]', '', text)
    return " ".join(text.split())

def donem_from_filename(filename):
    m = re.search(r'_(\d+)\._', filename)
    return int(m.group(1)) if m else None

def extract_from_pdf(pdf_path):
    donem = donem_from_filename(pdf_path.name)
    records = []
    current_il = ""
    current_parti = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                ncols = len(table[0]) if table[0] else 0
                if ncols < 4:
                    continue

                header = [clean(c) for c in (table[0] or [])]
                has_header = any(h in ("Milletvekili", "Seçim Bölgesi") for h in header)

                # column layout options:
                #   4 col: [il, isim, parti, degisiklik]
                #   5 col: [il, isim, spacer, parti, degisiklik]
                #   6 col: [il, isim, spacer, parti, spacer, degisiklik]
                col_il = 0
                col_isim = 1
                col_parti = 3 if ncols >= 5 else 2
                col_degisiklik = 5 if ncols >= 6 else (ncols - 1)

                # skip summary/sandalye dağılımı table (header contains "Seçilen" or "Dönem sonu")
                if any(h in ("Seçilen", "Dönem sonu", "Seçim") for h in header):
                    continue

                start_row = 1 if has_header else 0

                for row in table[start_row:]:
                    if not row:
                        continue
                    cells = [clean(c) for c in row]
                    while len(cells) <= col_degisiklik:
                        cells.append("")

                    il_cell = cells[col_il]
                    isim_cell = cells[col_isim]
                    parti_cell = cells[col_parti]
                    degisiklik_cell = cells[col_degisiklik]

                    # fill forward merged cells across pages
                    if il_cell:
                        current_il = il_cell
                    if parti_cell:
                        current_parti = parti_cell

                    isim = isim_cell
                    if not isim:
                        continue

                    records.append({
                        "donem": donem,
                        "il": current_il,
                        "isim": isim,
                        "parti": current_parti,
                        "degisiklik": degisiklik_cell
                    })

    return records

def main():
    all_records = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDF bulundu.")

    for pdf_path in pdfs:
        print(f"  İşleniyor: {pdf_path.name}")
        recs = extract_from_pdf(pdf_path)
        print(f"    -> {len(recs)} kayıt")
        all_records.extend(recs)

    print(f"\nToplam {len(all_records)} milletvekili kaydı çıkarıldı.")

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"JSON kaydedildi: {OUTPUT_JSON}")

    # CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["donem", "il", "isim", "parti", "degisiklik"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"CSV kaydedildi: {OUTPUT_CSV}")

    # quality check
    print("\n--- Örnek kayıtlar (ilk 10) ---")
    for r in all_records[:10]:
        print(r)

    from collections import Counter
    counts = Counter(r["donem"] for r in all_records)
    print("\n--- Dönem başına kayıt sayısı ---")
    for d in sorted(counts):
        print(f"  {d}. dönem: {counts[d]}")

    empty_il = [r for r in all_records if not r["il"]]
    empty_parti = [r for r in all_records if not r["parti"]]
    print(f"\nİl boş: {len(empty_il)}, Parti boş: {len(empty_parti)}")

if __name__ == "__main__":
    main()
