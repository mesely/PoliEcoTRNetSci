"""
annotate_candidates.py
──────────────────────
Terminal tabanlı annotation aracı.
Her kayıt sonrası CSV'ye kaydeder — çökme güvenli.

Kullanım: python3 annotate_candidates.py

Kısayollar
──────────
  n  → negative   (saldırı, eleştiri, muhalefet)
  p  → positive   (destek, övgü, işbirliği)
  m  → mixed      (hem + hem -, belirsiz)
  s  → skip       (okunaksız, konu dışı)
  q  → kaydet ve çık
"""

import os
import sys
import csv
import shutil
import textwrap
from pathlib import Path
from datetime import datetime

CSV_PATH = Path(__file__).parent / "data" / "annotation_candidates_v2.csv"

# ANSI renk kodları
BOLD  = "\033[1m"
RED   = "\033[91m"
GRN   = "\033[92m"
YLW   = "\033[93m"
BLU   = "\033[94m"
CYN   = "\033[96m"
RST   = "\033[0m"
DIM   = "\033[2m"

VALID = {"n": "negative", "p": "positive", "m": "mixed", "s": "skip", "q": "quit"}


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def bar(done, total, width=40):
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def show_candidate(row: dict, idx: int, total: int, labeled: int):
    clear()
    print(f"{BLU}{'─'*70}{RST}")
    print(f"{BOLD}  Annotation — {bar(labeled, total)}{RST}")
    print(f"{DIM}  Bu aday: {idx+1}/{total}  |  Etiketlenen: {labeled}{RST}")
    print(f"{BLU}{'─'*70}{RST}\n")

    # Metadata
    print(f"  {DIM}ID:{RST}           {row['candidate_id']}")
    print(f"  {DIM}Tarih:{RST}        {row.get('date','?')}  ({row.get('year','?')})")
    print(f"  {DIM}Konuşmacı:{RST}    {row.get('speaker','?')}")
    print(f"  {CYN}Kaynak parti:{RST} {row.get('source_party','?')}")
    print(f"  {CYN}Hedef parti:{RST}  {row.get('target_party','?')}")

    # BEME skoru
    score = float(row.get("heuristic_score", 0) or 0)
    pos_h = int(row.get("heuristic_positive_hits", 0) or 0)
    neg_h = int(row.get("heuristic_negative_hits", 0) or 0)
    h_lbl = row.get("heuristic_label", "?")
    lbl_color = GRN if h_lbl == "positive" else (RED if h_lbl == "negative" else YLW)
    print(f"  {DIM}BEME skoru:{RST}   {score:+.3f}  "
          f"(+hits:{pos_h}  -hits:{neg_h})  "
          f"→ {lbl_color}{h_lbl}{RST}")

    # Metin
    print(f"\n{BOLD}  Metin:{RST}")
    text = str(row.get("window_text", "")).strip()
    wrapped = textwrap.fill(text, width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    # Mevcut etiket
    cur = str(row.get("manual_label", "")).strip()
    if cur and cur != "skip":
        print(f"\n  {DIM}Mevcut etiket:{RST} {GRN if cur=='positive' else RED}{cur}{RST}")

    print(f"\n{BLU}{'─'*70}{RST}")
    print(f"  {GRN}p{RST}=positive  {RED}n{RST}=negative  {YLW}m{RST}=mixed  "
          f"{DIM}s{RST}=skip  {DIM}q{RST}=kaydet+çık")
    print(f"{BLU}{'─'*70}{RST}")


def load_rows():
    if not CSV_PATH.exists():
        print(f"Hata: {CSV_PATH} bulunamadı.")
        print("Önce çalıştır: python3 sample_annotation_candidates.py")
        sys.exit(1)
    rows = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    return rows, fieldnames


def save_rows(rows, fieldnames):
    tmp = CSV_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(str(tmp), str(CSV_PATH))


def main():
    rows, fieldnames = load_rows()
    total   = len(rows)
    labeled = sum(1 for r in rows if r.get("manual_label","").strip()
                  and r["manual_label"] != "skip")

    # Etiketlenmemiş (boş veya skip) index listesi
    queue = [i for i, r in enumerate(rows)
             if not r.get("manual_label","").strip() or r["manual_label"] == "skip"]

    if not queue:
        print(f"\n✅ Tüm {total} aday etiketlendi! Şimdi build_goldset_v2.py çalıştır.\n")
        return

    print(f"\n{BOLD}Annotation başlıyor — {len(queue)} aday bekliyor.{RST}")
    print(f"İpucu: herhangi bir zamanda 'q' yazarak çıkabilirsin.\n")
    input("Başlamak için Enter'a bas...")

    try:
        for pos, idx in enumerate(queue):
            row = rows[idx]
            show_candidate(row, pos, len(queue), labeled)

            while True:
                try:
                    raw = input("\n  Etiket: ").strip().lower()
                except EOFError:
                    raw = "q"

                if raw not in VALID:
                    print(f"  {RED}Geçersiz. {', '.join(VALID.keys())} kullan.{RST}")
                    continue

                if raw == "q":
                    save_rows(rows, fieldnames)
                    print(f"\n✅ Kaydedildi. {labeled}/{total} tamamlandı.")
                    return

                rows[idx]["manual_label"] = VALID[raw]
                rows[idx]["manual_notes"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                if raw != "s":
                    labeled += 1
                save_rows(rows, fieldnames)
                break

    except KeyboardInterrupt:
        save_rows(rows, fieldnames)
        print(f"\n\n✅ Kaydedildi. {labeled}/{total} tamamlandı.")
        return

    clear()
    print(f"\n{GRN}{'='*50}{RST}")
    print(f"  Tüm adaylar tamamlandı! 🎉")
    print(f"  Etiketlenen: {labeled}/{total}")
    print(f"  Sonraki adım: python3 build_goldset_v2.py")
    print(f"{GRN}{'='*50}{RST}\n")


if __name__ == "__main__":
    main()
