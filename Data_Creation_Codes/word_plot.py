from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.data import find
from wordcloud import WordCloud

from plot import DATA_PATH, NON_MP_ROLES, PARTY_COLORS


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
WORD_SAVE_DIR = PROJECT_ROOT / "TBMM_EDA_Figures" / "TBMM_Word_Plots"
WORD_SAVE_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_WORDS_CSV_PATH = WORD_SAVE_DIR / "TBMM_Tum_Kelime_Ozetleri.csv"
WORD_AUDIT_DIR = DATA_DIR / "word_audit"
PROPOSED_STOPWORDS_PATH = WORD_AUDIT_DIR / "proposed_stopwords.csv"
WORDCLOUD_SIZE = (1800, 1000)
VALID_DONEMLER = ["22", "23", "24", "25", "26", "27", "28"]
TOP_WORDS = 1000


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("i̇", "i")
        .replace("ı", "i")
        .replace("â", "a")
        .replace("î", "i")
        .replace("û", "u")
    )


def slugify(text: str) -> str:
    text = normalize_text(text).replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "."})


def party_color_map(parties: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    fallback = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    idx = 0
    for party in parties:
        if party in PARTY_COLORS:
            color_map[party] = PARTY_COLORS[party]
        else:
            color_map[party] = fallback[idx % len(fallback)]
            idx += 1
    return color_map


def tokenize(text: str) -> list[str]:
    import re

    return [token for token in re.findall(r"[a-zçğıöşüâîû]{3,}", normalize_text(text)) if token not in ALL_STOPWORDS]


def load_stopwords() -> set[str]:
    try:
        find("corpora/stopwords")
        base = {normalize_text(word) for word in stopwords.words("turkish")}
    except LookupError:
        base = set()

    extra = {
        "sayin", "baskan", "degerli", "milletvekilleri", "milletvekili", "tesekkur",
        "ediyorum", "ederim", "soz", "genel", "kurul", "meclis", "turkiye",
        "saygiyla", "selamliyorum", "efendim", "arkadaslar", "arkadaslarim",
        "arkadaslarimiz", "yilinda", "yili", "meclisin", "biliyor", "diyoruz",
        "tesekkurler", "katilmiyoruz",
    }

    if PROPOSED_STOPWORDS_PATH.exists():
        audit_df = pd.read_csv(PROPOSED_STOPWORDS_PATH)
        audit_words = set(
            audit_df[audit_df["recommendation"].isin({"add_stopword", "review_first"})]["word"]
            .astype(str)
            .map(normalize_text)
        )
    else:
        audit_words = set()

    return base.union(extra).union(audit_words)


ALL_STOPWORDS = load_stopwords()


def sample_for_words(df: pd.DataFrame, sample_size: int = 20000) -> pd.DataFrame:
    frames = []
    for donem in VALID_DONEMLER:
        chunk = df[df["donem"] == donem]
        if chunk.empty:
            continue
        frames.append(chunk.sample(n=min(sample_size, len(chunk)), random_state=42))
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()


def build_word_table(df: pd.DataFrame, group_col: str | None = None, top_k: int = TOP_WORDS) -> pd.DataFrame:
    from collections import Counter

    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_col) if group_col else [("ALL", df)]
    for key, chunk in grouped:
        counter = Counter()
        for speech in chunk["speech"].astype(str):
            counter.update(tokenize(speech))
        for rank, (word, freq) in enumerate(counter.most_common(top_k), start=1):
            row = {"rank": rank, "word": word, "freq": freq}
            if group_col:
                row[group_col] = key
            rows.append(row)
    return pd.DataFrame(rows)


def build_combined_words_csv(
    overall_words: pd.DataFrame,
    by_donem_words: pd.DataFrame,
    by_party_words: pd.DataFrame,
    by_donem_party_words: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    overall = overall_words.copy()
    overall["scope"] = "overall"
    overall["donem"] = ""
    overall["party"] = ""
    frames.append(overall[["scope", "donem", "party", "rank", "word", "freq"]])

    donem = by_donem_words.copy()
    donem["scope"] = "donem"
    donem["party"] = ""
    frames.append(donem[["scope", "donem", "party", "rank", "word", "freq"]])

    party = by_party_words.copy()
    party["scope"] = "party"
    party["donem"] = ""
    frames.append(party[["scope", "donem", "party", "rank", "word", "freq"]])

    donem_party = by_donem_party_words.copy()
    donem_party["scope"] = "donem_party"
    frames.append(donem_party[["scope", "donem", "party", "rank", "word", "freq"]])

    return pd.concat(frames, ignore_index=True)


def make_wordcloud(freq_df: pd.DataFrame, title: str, output_path: Path, color: str) -> None:
    if freq_df.empty:
        return

    frequencies = dict(freq_df[["word", "freq"]].itertuples(index=False, name=None))
    wc = WordCloud(
        width=WORDCLOUD_SIZE[0],
        height=WORDCLOUD_SIZE[1],
        background_color="white",
        max_words=min(len(frequencies), 160),
        prefer_horizontal=0.88,
        collocations=False,
        random_state=42,
        min_font_size=10,
        max_font_size=170,
        color_func=lambda *args, **kwargs: color,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    print("📂 Veri seti yükleniyor...")
    df = pd.read_csv(DATA_PATH)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["speech"] = df["speech"].fillna("").astype(str)

    df_mps = df[~df["party"].isin(NON_MP_ROLES)].copy()
    color_map = party_color_map(sorted(df_mps["party"].dropna().unique()))

    print("🧪 Kelime tabloları hazırlanıyor...")
    sampled_words = sample_for_words(df_mps)
    overall_words = build_word_table(sampled_words, None, TOP_WORDS)
    by_donem_words = build_word_table(sampled_words, "donem", TOP_WORDS)
    by_party_words = build_word_table(sampled_words, "party", TOP_WORDS)

    donem_party_rows = []
    for (donem, party), chunk in sampled_words.groupby(["donem", "party"]):
        for rank, (word, freq) in enumerate(build_word_table(chunk, None, TOP_WORDS)[["word", "freq"]].itertuples(index=False, name=None), start=1):
            donem_party_rows.append(
                {
                    "donem": donem,
                    "party": party,
                    "rank": rank,
                    "word": word,
                    "freq": freq,
                }
            )
    by_donem_party_words = pd.DataFrame(donem_party_rows)

    print("☁️ Word cloud görselleri çiziliyor...")
    make_wordcloud(
        overall_words.head(140),
        "Meclis Geneli Kelime Bulutu",
        WORD_SAVE_DIR / "genel_kelime_alani.png",
        "#2F4858",
    )

    for donem in VALID_DONEMLER:
        donem_df = by_donem_words[by_donem_words["donem"] == donem].head(140)
        if donem_df.empty:
            continue

        print(f"🎬 {donem}. dönem kelime bulutu hazırlanıyor...")
        make_wordcloud(
            donem_df,
            f"{donem}. Dönem Kelime Bulutu",
            WORD_SAVE_DIR / f"{donem}_kelime_alani.png",
            "#3A506B",
        )

    for party in sorted(df_mps["party"].dropna().unique()):
        party_words = by_party_words[by_party_words["party"] == party].head(120)
        if party_words.empty:
            continue

        party_slug = slugify(party)
        print(f"🎨 {party} için kelime bulutu hazırlanıyor...")
        make_wordcloud(
            party_words,
            f"{party} | Kelime Bulutu",
            WORD_SAVE_DIR / f"{party_slug}_kelime_alani.png",
            color_map.get(party, "#666666"),
        )

    for donem in VALID_DONEMLER:
        donem_party_df = by_donem_party_words[by_donem_party_words["donem"] == donem]
        if donem_party_df.empty:
            continue

        for party in sorted(donem_party_df["party"].dropna().unique()):
            freq_df = donem_party_df[donem_party_df["party"] == party].head(120)
            if len(freq_df) < 20:
                continue

            party_slug = slugify(party)
            print(f"🗂️ {donem} | {party} için dönem-parti bulutu hazırlanıyor...")
            make_wordcloud(
                freq_df,
                f"{donem}. Dönem | {party} | Kelime Bulutu",
                WORD_SAVE_DIR / f"{donem}_{party_slug}_kelime_alani.png",
                color_map.get(party, "#666666"),
            )

    print("💾 Kelime CSV çıktıları yazılıyor...")
    overall_words.to_csv(WORD_SAVE_DIR / "overall_top_1000_words.csv", index=False, encoding="utf-8-sig")
    by_donem_words.to_csv(WORD_SAVE_DIR / "top_1000_words_by_donem.csv", index=False, encoding="utf-8-sig")
    by_party_words.to_csv(WORD_SAVE_DIR / "top_1000_words_by_party.csv", index=False, encoding="utf-8-sig")
    by_donem_party_words.to_csv(WORD_SAVE_DIR / "top_1000_words_by_donem_party.csv", index=False, encoding="utf-8-sig")

    combined_words = build_combined_words_csv(overall_words, by_donem_words, by_party_words, by_donem_party_words)
    combined_words.to_csv(COMBINED_WORDS_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ Kelime görselleri ve özet CSV hazır: {WORD_SAVE_DIR}")
    print(f"🧾 Birleşik CSV: {COMBINED_WORDS_CSV_PATH}")


if __name__ == "__main__":
    main()
