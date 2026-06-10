from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
DATA_PATH_CANDIDATES = [
    DATA_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    DATA_DIR / "TBMM_Network_Dataset.csv",
    ROOT_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    ROOT_DIR / "TBMM_Network_Dataset.csv",
]
DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), DATA_PATH_CANDIDATES[0])
OUTPUT_DIR = DATA_DIR / "word_audit"
VALID_DONEMLER = ["22", "23", "24", "25", "26", "27", "28"]
NON_MP_ROLES = {"MECLİS BAŞKANLIĞI", "KOMİSYON SÖZCÜSÜ", "Bilinmeyen", "BAKANLAR KURULU"}
TOKEN_RE = re.compile(r"[a-zçğıöşüâîû]{2,}")
SUFFIX_LIKE_RE = re.compile(r"^(nin|nin|nun|nün|nde|nda|den|dan|yle|yla|nci|inci|uncu|üncü|dır|dir|dur|dür|tir|tır|tur|tür)$")

BASE_STOPWORDS = {
    "sayın", "başkan", "değerli", "milletvekilleri", "bir", "bu", "da", "de", "için", "ne", "o", "ve",
    "ile", "veya", "ise", "şu", "biz", "yok", "var", "türkiye", "meclis", "yüce", "karar", "genel",
    "kurul", "teşekkür", "ediyorum", "arz", "ederim", "söz", "konusu", "birleşim", "oturum", "saat",
    "açıyorum", "açılmıştır", "teklif", "kabul", "reddedilmiştir", "efendim", "sunuyorum", "oylarınıza",
    "sunulmuştur", "oylama", "yapılacaktır", "katılmıyoruz", "uygun", "görülmüştür", "şimdi", "kendi",
    "neden", "nasıl", "burada", "şöyle", "böyle", "onlar", "şunlar", "tutanak", "maddesine",
    "geçilmesi", "reddedenler", "buyurun", "temsilen", "başarılar", "dilerim", "hayırlı", "olsun",
    "diliyorum", "sıralarından", "alkışlar", "gürültüler", "büyük", "ilgili", "bunu", "yüzde", "parti",
    "değil", "hayır", "evet", "sesleri", "laf", "atmalar", "şunları", "bunları", "onları", "birçok",
    "biraz", "başka", "zaman", "yoktur", "vardır", "gerekmektedir", "lazımdır", "gerekiyor", "yoksa",
    "devamla", "arkadaşlar", "görüşülmekte", "olan", "gibi", "olarak", "edenler", "etmeyenler",
    "edilmiştir", "görüş", "önerisi", "hükümet", "bölüm", "üzerinde", "başkanım", "saygıyla",
    "selamlıyorum", "kanun", "tasarısı", "önerge", "otomatik", "mikrofon", "cihaz", "kapatıldı",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-per-donem", type=int, default=18000)
    parser.add_argument("--top-k", type=int, default=1500)
    parser.add_argument("--party-top-k", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-non-mp", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("i̇", "i")
        .replace("ı", "i")
        .replace("â", "a")
        .replace("î", "i")
        .replace("û", "u")
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def sample_corpus(df: pd.DataFrame, sample_per_donem: int, random_state: int) -> pd.DataFrame:
    chunks = []
    for donem in VALID_DONEMLER:
        block = df[df["donem"] == donem]
        if block.empty:
            continue
        chunks.append(block.sample(n=min(sample_per_donem, len(block)), random_state=random_state))
    return pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()


def category_for(word: str, term_freq: int, doc_ratio: float, donem_span: int, party_span: int) -> str:
    if SUFFIX_LIKE_RE.match(word):
        return "suffix_fragment"
    if word in BASE_STOPWORDS:
        return "existing_stopword"
    if word in {"sayin", "baskanim", "selamliyorum", "saygiyla", "alkislar", "siralarindan"}:
        return "parliamentary_procedure"
    if word in {"otomatik", "mikrofon", "cihaz", "kapatildi"}:
        return "transcript_artifact"
    if doc_ratio >= 0.09 and donem_span == 7:
        return "generic_high_coverage"
    if party_span >= 10 and doc_ratio >= 0.04:
        return "cross_party_generic"
    if term_freq >= 4000 and donem_span == 7:
        return "high_frequency_generic"
    return "content_or_borderline"


def recommendation_for(category: str) -> str:
    mapping = {
        "suffix_fragment": "add_stopword",
        "existing_stopword": "already_present",
        "parliamentary_procedure": "add_stopword",
        "transcript_artifact": "add_stopword",
        "generic_high_coverage": "review_first",
        "cross_party_generic": "review_first",
        "high_frequency_generic": "review_first",
        "content_or_borderline": "keep_for_now",
    }
    return mapping[category]


def build_global_counts(df: pd.DataFrame) -> tuple[Counter, Counter, Counter, Counter]:
    tf_counter = Counter()
    doc_counter = Counter()
    donem_counter = Counter()
    party_counter = Counter()

    for _, row in df[["speech", "donem", "party"]].iterrows():
        tokens = tokenize(str(row["speech"]))
        if not tokens:
            continue
        token_set = set(tokens)
        tf_counter.update(tokens)
        doc_counter.update(token_set)
        for token in token_set:
            donem_counter[(token, row["donem"])] += 1
            party_counter[(token, row["party"])] += 1

    return tf_counter, doc_counter, donem_counter, party_counter


def build_candidate_table(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    tf_counter, doc_counter, donem_counter, party_counter = build_global_counts(df)
    total_docs = len(df)
    rows = []

    for word, term_freq in tf_counter.most_common(top_k):
        doc_freq = doc_counter[word]
        doc_ratio = doc_freq / total_docs if total_docs else 0
        donem_span = sum(1 for donem in VALID_DONEMLER if donem_counter[(word, donem)] > 0)
        party_span = len({party for (token, party), value in party_counter.items() if token == word and value > 0})
        category = category_for(word, term_freq, doc_ratio, donem_span, party_span)
        recommendation = recommendation_for(category)
        rows.append(
            {
                "word": word,
                "term_freq": term_freq,
                "doc_freq": doc_freq,
                "doc_ratio": round(doc_ratio, 6),
                "donem_span": donem_span,
                "party_span": party_span,
                "already_stopword": word in BASE_STOPWORDS,
                "category": category,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)


def build_group_top_words(df: pd.DataFrame, group_col: str, top_k: int) -> pd.DataFrame:
    rows = []
    for group_key, chunk in df.groupby(group_col):
        counter = Counter()
        for speech in chunk["speech"].astype(str):
            counter.update(tokenize(speech))
        for rank, (word, freq) in enumerate(counter.most_common(top_k), start=1):
            rows.append({group_col: group_key, "rank": rank, "word": word, "freq": freq})
    return pd.DataFrame(rows)


def build_proposed_stopwords(candidate_df: pd.DataFrame) -> pd.DataFrame:
    selected = candidate_df[candidate_df["recommendation"].isin({"add_stopword", "review_first"})].copy()
    selected["priority"] = selected["recommendation"].map({"add_stopword": 1, "review_first": 2})
    return selected.sort_values(
        ["priority", "term_freq", "doc_freq"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_summary(df_full: pd.DataFrame, df_sampled: pd.DataFrame, candidate_df: pd.DataFrame) -> str:
    counts = df_sampled["donem"].value_counts().sort_index()
    selected = candidate_df[candidate_df["recommendation"].isin({"add_stopword", "review_first"})]
    lines = [
        "Stopword Audit Summary",
        "----------------------",
        f"full_rows: {len(df_full)}",
        f"sampled_rows: {len(df_sampled)}",
        "sample_by_donem:",
    ]
    lines.extend(f"{donem}: {count}" for donem, count in counts.items())
    lines.extend(
        [
            "",
            f"candidate_rows: {len(candidate_df)}",
            f"selected_for_review_or_add: {len(selected)}",
            "",
            "output_files:",
            "- candidate_stopwords_detailed.csv",
            "- proposed_stopwords.csv",
            "- top_words_by_donem.csv",
            "- top_words_by_party.csv",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    print("📂 Veri seti yukleniyor...")
    df = pd.read_csv(args.data_path)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["speech"] = df["speech"].fillna("").astype(str)

    if not args.include_non_mp:
        df = df[~df["party"].isin(NON_MP_ROLES)].copy()

    sampled = sample_corpus(df, args.sample_per_donem, args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("🧪 Stopword adaylari hesaplanıyor...")
    candidate_df = build_candidate_table(sampled, args.top_k)
    proposed_df = build_proposed_stopwords(candidate_df)
    by_donem_df = build_group_top_words(sampled, "donem", args.top_k)
    major_parties = sampled["party"].value_counts().head(args.party_top_k).index
    by_party_df = build_group_top_words(sampled[sampled["party"].isin(major_parties)], "party", args.top_k)

    candidate_df.to_csv(args.output_dir / "candidate_stopwords_detailed.csv", index=False, encoding="utf-8-sig")
    proposed_df.to_csv(args.output_dir / "proposed_stopwords.csv", index=False, encoding="utf-8-sig")
    by_donem_df.to_csv(args.output_dir / "top_words_by_donem.csv", index=False, encoding="utf-8-sig")
    by_party_df.to_csv(args.output_dir / "top_words_by_party.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.txt").write_text(
        build_summary(df, sampled, candidate_df),
        encoding="utf-8",
    )

    print(f"✅ Stopword audit tamam: {args.output_dir}")


if __name__ == "__main__":
    main()
