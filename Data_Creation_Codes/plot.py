from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
SAVE_DIR = PROJECT_ROOT / "TBMM_EDA_Figures" / "TBMM_Plots"
DATA_PATH_CANDIDATES = [
    DATA_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    DATA_DIR / "TBMM_Network_Dataset.csv",
    ROOT_DIR / "TBMM_Network_Dataset_partyfixed.csv",
    ROOT_DIR / "TBMM_Network_Dataset.csv",
]
DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), DATA_PATH_CANDIDATES[0])
VALID_DONEMLER = ["22", "23", "24", "25", "26", "27", "28"]
NON_MP_ROLES = {"MECLİS BAŞKANLIĞI", "KOMİSYON SÖZCÜSÜ", "Bilinmeyen", "BAKANLAR KURULU"}

SAVE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font="DejaVu Sans")

PARTY_COLORS = {
    "Adalet ve Kalkınma Partisi": "#F4B400",
    "AK PARTİ": "#F4B400",
    "AKP": "#F4B400",
    "Cumhuriyet Halk Partisi": "#D7263D",
    "CHP": "#D7263D",
    "Milliyetçi Hareket Partisi": "#A61E4D",
    "MHP": "#A61E4D",
    "Halkların Demokratik Partisi": "#6A1B9A",
    "HDP": "#6A1B9A",
    "DEM Parti": "#6A1B9A",
    "Halkların Eşitlik ve Demokrasi Partisi": "#6A1B9A",
    "İYİ Parti": "#2D7DD2",
    "DEVA Partisi": "#1F77B4",
    "Demokrasi ve Atılım Partisi": "#1F77B4",
    "Gelecek Partisi": "#2E8B57",
    "Saadet Partisi": "#C8102E",
    "YENİ YOL Partisi": "#3A86FF",
    "Anavatan Partisi": "#FFD23F",
    "Doğru Yol Partisi": "#0B4F6C",
    "Demokrat Parti": "#4C78A8",
    "Demokratik Sol Parti": "#4EA8DE",
    "Türkiye İşçi Partisi": "#B56576",
    "Yeniden Refah Partisi": "#2A9D8F",
    "Bağımsız": "#6C757D",
}


def party_color_map(parties: list[str]) -> dict[str, str]:
    fallback_palette = sns.color_palette("tab20", n_colors=max(len(parties), 3))
    color_map: dict[str, str] = {}
    fallback_idx = 0
    for party in parties:
        if party in PARTY_COLORS:
            color_map[party] = PARTY_COLORS[party]
        else:
            color_map[party] = mcolors.to_hex(fallback_palette[fallback_idx % len(fallback_palette)])
            fallback_idx += 1
    return color_map


def plot_stacked_bar(df: pd.DataFrame, value_col: str, title: str, output_path: Path, color_map: dict[str, str]) -> None:
    if df.empty:
        return
    pivot = (
        df.pivot_table(index="donem", columns="party", values=value_col, aggfunc="sum", fill_value=0)
        .reindex(VALID_DONEMLER)
        .fillna(0)
    )
    top_parties = pivot.sum(axis=0).sort_values(ascending=False).head(10).index.tolist()
    pivot = pivot[top_parties]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(16, 9),
        color=[color_map.get(party, "#999999") for party in pivot.columns],
        width=0.82,
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Dönem")
    ax.set_ylabel("Toplam")
    ax.legend(title="Parti", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_top_speakers(series: pd.Series, title: str, output_path: Path, colors: list[str]) -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(8, len(series) * 0.45)))
    sns.barplot(x=series.values, y=series.index, color="#2F4858", ax=ax)
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Konuşma Sayısı")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_party_power_dual(donem_df: pd.DataFrame, donem: str, color_map: dict[str, str]) -> None:
    if donem_df.empty:
        return

    unique_speakers = (
        donem_df.groupby("party")["speaker"]
        .nunique()
        .sort_values(ascending=False)
    )
    avg_speeches = (
        donem_df.groupby("party")
        .size()
        .div(donem_df.groupby("party")["speaker"].nunique())
        .sort_values(ascending=False)
    )

    ordered_parties = unique_speakers.index.tolist()
    avg_speeches = avg_speeches.reindex(avg_speeches.index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    sns.barplot(
        x=unique_speakers.values,
        y=unique_speakers.index,
        hue=unique_speakers.index,
        palette=[color_map.get(party, "#999999") for party in unique_speakers.index],
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(f"{donem}. Dönem | Benzersiz Milletvekili Sayısı", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Milletvekili Sayısı")
    axes[0].set_ylabel("")

    sns.barplot(
        x=avg_speeches.values,
        y=avg_speeches.index,
        hue=avg_speeches.index,
        palette=[color_map.get(party, "#999999") for party in avg_speeches.index],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title(f"{donem}. Dönem | Kişi Başına Ortalama Konuşma", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Ortalama Konuşma")
    axes[1].set_ylabel("")

    fig.suptitle(f"{donem}. Dönem Parti Güç Dağılımı", fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"{donem}_parti_guc_dagilimi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("📂 Veri seti yükleniyor...")
    df = pd.read_csv(DATA_PATH)
    df["donem"] = df["donem"].astype(str).str.replace(".0", "", regex=False)
    df["speech"] = df["speech"].fillna("").astype(str)

    df_mps = df[~df["party"].isin(NON_MP_ROLES)].copy()
    color_map = party_color_map(sorted(df_mps["party"].dropna().unique()))

    print("🎨 Dönemler arası kompozisyon grafikleri hazırlanıyor...")
    speech_summary = (
        df_mps.groupby(["donem", "party"], as_index=False)
        .size()
        .rename(columns={"size": "speech_count"})
    )
    speaker_summary = (
        df_mps.groupby(["donem", "party"], as_index=False)["speaker"]
        .nunique()
        .rename(columns={"speaker": "speaker_count"})
    )

    plot_stacked_bar(
        speech_summary,
        "speech_count",
        "Dönemlere Göre Parti Bazlı Toplam Konuşma Hacmi",
        SAVE_DIR / "donem_parti_konusma_stacked.png",
        color_map,
    )
    plot_stacked_bar(
        speaker_summary,
        "speaker_count",
        "Dönemlere Göre Parti Bazlı Benzersiz Konuşmacı Sayısı",
        SAVE_DIR / "donem_parti_konusmaci_stacked.png",
        color_map,
    )

    for donem in VALID_DONEMLER:
        donem_df = df_mps[df_mps["donem"] == donem].copy()
        if donem_df.empty:
            continue

        print(f"🎬 {donem}. dönem işleniyor...")
        plot_party_power_dual(donem_df, donem, color_map)

        top_speakers = donem_df["speaker"].value_counts().head(20)
        speaker_party_map = (
            donem_df.groupby("speaker")["party"]
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )
        speaker_colors = [
            color_map.get(speaker_party_map.get(speaker, ""), "#2F4858")
            for speaker in top_speakers.index
        ]
        plot_top_speakers(
            top_speakers,
            f"{donem}. Dönem En Aktif 20 Milletvekili",
            SAVE_DIR / f"{donem}_en_aktif_vekiller.png",
            speaker_colors,
        )

        major_parties = donem_df["party"].value_counts().head(8).index
        for party in major_parties:
            party_df = donem_df[donem_df["party"] == party]
            if len(party_df) < 20:
                continue
            party_slug = party.lower().replace(" ", "_").replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c")
            top_party_speakers = party_df["speaker"].value_counts().head(15)
            plot_top_speakers(
                top_party_speakers,
                f"{donem}. Dönem | {party} | En Aktif 15 İsim",
                SAVE_DIR / f"{donem}_{party_slug}_en_aktifler.png",
                [color_map.get(party, "#666666")] * len(top_party_speakers),
            )

    print(f"✅ EDA tamamlandı. Çıktılar: {SAVE_DIR}")


if __name__ == "__main__":
    main()
