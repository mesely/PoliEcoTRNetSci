from __future__ import annotations

import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECONOMY_ROOT = PROJECT_ROOT / "economy_data"
RAW_DIR = ECONOMY_ROOT / "raw"
PREPROCESSED_DIR = ECONOMY_ROOT / "preprocessed"

MANUAL_UPLOADS_DIR = RAW_DIR / "manual_uploads"
API_DOWNLOADS_DIR = RAW_DIR / "api_downloads"
LEGACY_GENERATED_DIR = RAW_DIR / "legacy_generated"

DATE_DDMMYYYY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
DATE_YYYYMM_RE = re.compile(r"^\d{4}-\d{2}$")
DATE_YYYYMM_COMBINED_RE = re.compile(r"^(?P<date>\d{4}-\d{2})\s+(?P<value>-?\d+(?:\.\d+)?)$")

TERM_WINDOWS = (
    {"term": 22, "start": pd.Timestamp("2002-11-01"), "end": pd.Timestamp("2007-07-31")},
    {"term": 23, "start": pd.Timestamp("2007-08-01"), "end": pd.Timestamp("2011-06-30")},
    {"term": 24, "start": pd.Timestamp("2011-07-01"), "end": pd.Timestamp("2015-06-30")},
    {"term": 25, "start": pd.Timestamp("2015-07-01"), "end": pd.Timestamp("2015-11-30")},
    {"term": 26, "start": pd.Timestamp("2015-12-01"), "end": pd.Timestamp("2018-06-30")},
    {"term": 27, "start": pd.Timestamp("2018-07-01"), "end": pd.Timestamp("2023-05-31")},
    {"term": 28, "start": pd.Timestamp("2023-06-01"), "end": pd.Timestamp("2026-12-31")},
)

OOXML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

MONTH_VALUE_COLUMNS = [
    "market_days",
    "usd_try_avg",
    "usd_try_last",
    "usd_try_return_month_pct",
    "eur_try_avg",
    "eur_try_last",
    "eur_try_return_month_pct",
    "bist100_avg",
    "bist100_last",
    "bist100_return_month_pct",
    "gold_usd_avg",
    "gold_usd_last",
    "gold_usd_return_month_pct",
    "brent_oil_usd_avg",
    "brent_oil_usd_last",
    "brent_oil_usd_return_month_pct",
    "wti_oil_usd_avg",
    "wti_oil_usd_last",
    "wti_oil_usd_return_month_pct",
    "short_term_interest_rate_percent",
    "discount_rate_percent",
    "immediate_interbank_rate_percent",
    "cpi_all_items_index_fred",
    "cpi_inflation_yoy_percent_fred",
    "cpi_inflation_mom_percent_fred",
    "harmonized_unemployment_rate_percent",
    "industrial_production_index",
    "exports_value_fred",
    "imports_value_fred",
    "usd_try_tcmb_mid_avg",
    "usd_try_tcmb_mid_last",
    "eur_try_tcmb_mid_avg",
    "eur_try_tcmb_mid_last",
    "usd_try_avg_harmonized",
    "usd_try_last_harmonized",
    "eur_try_avg_harmonized",
    "eur_try_last_harmonized",
    "policy_rate_percent_tcmb",
    "gross_reserves_usd_million_tcmb",
    "m2_money_supply_try_avg",
    "m2_money_supply_try_last",
    "cpi_index_1994_100_tuik",
    "cpi_index_2025_100_tuik",
    "cpi_mom_percent_tuik",
    "cpi_yoy_percent_tuik",
    "cpi_12m_avg_change_percent_tuik",
    "cpi_vs_prev_dec_percent_tuik",
    "inflation_mom_percent_harmonized",
    "inflation_yoy_percent_harmonized",
    "industrial_production_index_tuik",
    "industrial_production_index_harmonized",
]


def ensure_dirs() -> None:
    MANUAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    API_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required file. Checked: {candidates}")


def sanitize_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not read CSV: {path}")


def col_to_num(label: str) -> int:
    value = 0
    for char in label:
        if char.isalpha():
            value = value * 26 + (ord(char.upper()) - 64)
    return value


def parse_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for node in root.findall("a:si", OOXML_NS):
        values.append("".join(text.text or "" for text in node.iterfind(".//a:t", OOXML_NS)))
    return values


def first_sheet_target(zip_file: ZipFile) -> str:
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheet = workbook.find("a:sheets", OOXML_NS)[0]
    rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    target = rel_map[rel_id]
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def read_ooxml_first_sheet(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zip_file:
        shared_strings = parse_shared_strings(zip_file)
        sheet_target = first_sheet_target(zip_file)
        root = ET.fromstring(zip_file.read(sheet_target))
        rows: list[list[str]] = []
        for row in root.iterfind(".//a:sheetData/a:row", OOXML_NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", OOXML_NS):
                reference = cell.attrib.get("r", "")
                index = col_to_num("".join(char for char in reference if char.isalpha()))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", OOXML_NS)
                inline_node = cell.find("a:is", OOXML_NS)
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(text.text or "" for text in inline_node.iterfind(".//a:t", OOXML_NS))
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                values[index] = value
            if values:
                max_index = max(values)
                rows.append([values.get(i, "") for i in range(1, max_index + 1)])
    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = []
    for idx, value in enumerate(padded_rows[0], start=1):
        cleaned = str(value).strip() or f"column_{idx}"
        header.append(cleaned)
    frame = pd.DataFrame(padded_rows[1:], columns=header)
    return frame.replace({"": pd.NA})


def parse_date_mixed(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    if DATE_DDMMYYYY_RE.match(text):
        return pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")
    if DATE_YYYYMM_RE.match(text):
        return pd.to_datetime(f"{text}-01", format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def pick_first_available(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"Missing expected columns: {candidates}")


def normalize_numeric_columns(frame: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column in exclude:
            continue
        output[column] = to_numeric(output[column])
    return output


def aggregate_observations_to_month(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    data = frame.dropna(subset=["date"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["date"])
    data["date"] = pd.to_datetime(data["date"])
    data["month_date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    grouped = data.sort_values("date").groupby("month_date")
    avg = grouped[value_columns].mean().add_suffix("_avg")
    last = grouped[value_columns].last().add_suffix("_last")
    monthly = avg.join(last).reset_index().rename(columns={"month_date": "date"})
    return monthly.sort_values("date").reset_index(drop=True)


def parse_manual_fx() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_existing(MANUAL_UPLOADS_DIR / "dolar_euro.xlsx", ECONOMY_ROOT / "dolar_euro.xlsx")
    frame = read_ooxml_first_sheet(path)
    date_col = pick_first_available(frame, ["Tarih", "Tarih "])
    frame = frame[frame[date_col].astype(str).str.fullmatch(r"\d{2}-\d{2}-\d{4}")].copy()
    frame["date"] = pd.to_datetime(frame[date_col], format="%d-%m-%Y", errors="coerce")
    frame["usd_try_bid_tcmb"] = to_numeric(frame["TP_DK_USD_A_YTL"])
    frame["usd_try_ask_tcmb"] = to_numeric(frame["TP_DK_USD_S_YTL"])
    frame["eur_try_bid_tcmb"] = to_numeric(frame["TP_DK_EUR_A_YTL"])
    frame["eur_try_ask_tcmb"] = to_numeric(frame["TP_DK_EUR_S_YTL"])
    frame["usd_try_tcmb_mid"] = frame[["usd_try_bid_tcmb", "usd_try_ask_tcmb"]].mean(axis=1, skipna=True)
    frame["eur_try_tcmb_mid"] = frame[["eur_try_bid_tcmb", "eur_try_ask_tcmb"]].mean(axis=1, skipna=True)
    daily = frame[
        [
            "date",
            "usd_try_bid_tcmb",
            "usd_try_ask_tcmb",
            "usd_try_tcmb_mid",
            "eur_try_bid_tcmb",
            "eur_try_ask_tcmb",
            "eur_try_tcmb_mid",
        ]
    ].sort_values("date")
    monthly = aggregate_observations_to_month(daily, ["usd_try_tcmb_mid", "eur_try_tcmb_mid"])
    return daily.reset_index(drop=True), monthly


def parse_policy_rate() -> pd.DataFrame:
    recent_path = find_existing(MANUAL_UPLOADS_DIR / "politika_faizi_1.xlsx", ECONOMY_ROOT / "politika_faizi_1.xlsx")
    older_path = find_existing(MANUAL_UPLOADS_DIR / "politika_faizi_2.xlsx", ECONOMY_ROOT / "politika_faizi_2.xlsx")

    recent = read_ooxml_first_sheet(recent_path)
    recent_date_col = pick_first_available(recent, ["Tarih", "Tarih "])
    recent = recent[recent[recent_date_col].astype(str).str.fullmatch(r"\d{4}-\d{2}")].copy()
    recent["date"] = pd.to_datetime(recent[recent_date_col] + "-01", format="%Y-%m-%d", errors="coerce")
    recent["policy_rate_percent_tcmb"] = to_numeric(recent["TP_BISPOLFAIZ_TUR"])
    recent = recent[["date", "policy_rate_percent_tcmb"]]

    older = read_ooxml_first_sheet(older_path)
    older_date_col = pick_first_available(older, ["Tarih", "Tarih "])
    combined = older[older_date_col].astype(str).str.extract(DATE_YYYYMM_COMBINED_RE)
    older = older.assign(date_raw=combined["date"], value_raw=combined["value"])
    older = older.dropna(subset=["date_raw"]).copy()
    older["date"] = pd.to_datetime(older["date_raw"] + "-01", format="%Y-%m-%d", errors="coerce")
    older["policy_rate_percent_tcmb"] = pd.to_numeric(older["value_raw"], errors="coerce")
    older = older[["date", "policy_rate_percent_tcmb"]]

    combined_rates = (
        pd.concat([older, recent], ignore_index=True)
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return combined_rates


def parse_reserves() -> pd.DataFrame:
    path = find_existing(MANUAL_UPLOADS_DIR / "uluslararasi_rezerv.xlsx", ECONOMY_ROOT / "uluslararasi_rezerv.xlsx")
    frame = read_ooxml_first_sheet(path)
    date_col = pick_first_available(frame, ["Tarih", "Tarih "])
    frame = frame[frame[date_col].astype(str).str.fullmatch(r"\d{4}-\d{2}")].copy()
    frame["date"] = pd.to_datetime(frame[date_col] + "-01", format="%Y-%m-%d", errors="coerce")
    frame["gross_reserves_usd_million_tcmb"] = to_numeric(frame["TP_REZVARPD_K1"])
    return frame[["date", "gross_reserves_usd_million_tcmb"]].sort_values("date").reset_index(drop=True)


def parse_m2() -> pd.DataFrame:
    path = find_existing(MANUAL_UPLOADS_DIR / "m2_para_arzi.xlsx", ECONOMY_ROOT / "m2_para_arzi.xlsx")
    frame = read_ooxml_first_sheet(path)
    date_col = pick_first_available(frame, ["Tarih", "Tarih "])
    frame = frame[frame[date_col].astype(str).str.fullmatch(r"\d{2}-\d{2}-\d{4}")].copy()
    frame["date"] = pd.to_datetime(frame[date_col], format="%d-%m-%Y", errors="coerce")
    frame["m2_money_supply_try"] = to_numeric(frame["TP_HPBITABLO1_11"])
    return aggregate_observations_to_month(frame[["date", "m2_money_supply_try"]], ["m2_money_supply_try"])


def parse_weekly_evds_file(filename: str) -> pd.DataFrame:
    path = find_existing(MANUAL_UPLOADS_DIR / filename, ECONOMY_ROOT / filename)
    frame = read_ooxml_first_sheet(path)
    date_col = pick_first_available(frame, ["Tarih", "Tarih "])
    frame = frame[frame[date_col].astype(str).str.fullmatch(r"\d{2}-\d{2}-\d{4}")].copy()
    frame["date"] = pd.to_datetime(frame[date_col], format="%d-%m-%Y", errors="coerce")
    value_columns = [column for column in frame.columns if column not in {date_col, "date"}]
    rename_map = {column: sanitize_slug(column) for column in value_columns}
    frame = frame.rename(columns=rename_map)
    frame = normalize_numeric_columns(frame, exclude={date_col, "date"})
    numeric_columns = [rename_map[column] for column in value_columns]
    monthly = aggregate_observations_to_month(frame[["date", *numeric_columns]], numeric_columns)
    return monthly


def parse_tufe_matrix(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path)
    month_columns = list(frame.columns[3:15])
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        series_label = str(row.iloc[0]).strip()
        metric_label = str(row.iloc[1]).strip()
        year_text = str(row.iloc[2]).strip()
        if not year_text or year_text.lower() == "nan":
            continue
        try:
            year = int(float(year_text))
        except ValueError:
            continue
        if series_label == "Satırlar":
            continue
        for month_index, column in enumerate(month_columns, start=1):
            value = pd.to_numeric(str(row[column]).replace(",", "."), errors="coerce")
            if pd.isna(value):
                continue
            records.append(
                {
                    "date": pd.Timestamp(year=year, month=month_index, day=1),
                    "series_label": series_label,
                    "metric_label": metric_label,
                    "value": value,
                }
            )
    return pd.DataFrame(records)


def parse_tufe() -> pd.DataFrame:
    older_path = find_existing(MANUAL_UPLOADS_DIR / "2002_2017_TUFE.csv", ECONOMY_ROOT / "2002_2017_TUFE.csv")
    newer_mom_path = find_existing(
        MANUAL_UPLOADS_DIR / "2017_2026_AYLIK_TUFE.csv",
        ECONOMY_ROOT / "2017_2026_AYLIK_TUFE.csv",
    )
    newer_index_path = find_existing(
        MANUAL_UPLOADS_DIR / "2017_2026_TUFE_GENEL.csv",
        ECONOMY_ROOT / "2017_2026_TUFE_GENEL.csv",
    )

    older = parse_tufe_matrix(older_path)
    newer_mom = parse_tufe_matrix(newer_mom_path)
    newer_index = parse_tufe_matrix(newer_index_path)

    monthly = pd.DataFrame({"date": pd.date_range("2002-01-01", "2026-12-01", freq="MS")})

    older_index = older[
        older["series_label"].eq("Tüketici Fiyatları Endeksi (1994=100)")
        & older["metric_label"].eq("Ölçüm bazında")
    ][["date", "value"]].rename(columns={"value": "cpi_index_1994_100_tuik"})

    older_mom = older[
        older["series_label"].str.contains("Aylık Değişim", regex=False)
    ][["date", "value"]].rename(columns={"value": "cpi_mom_percent_tuik"})

    older_yoy = older[
        older["series_label"].str.contains("Yıllık Değişim", regex=False)
    ][["date", "value"]].rename(columns={"value": "cpi_yoy_percent_tuik"})

    older_prev_dec = older[
        older["series_label"].str.contains("Aralık Ayına Göre Değişim", regex=False)
    ][["date", "value"]].rename(columns={"value": "cpi_vs_prev_dec_percent_tuik"})

    older_rolling = older[
        older["series_label"].str.contains("Hareketli Ortalamalara", regex=False)
    ][["date", "value"]].rename(columns={"value": "cpi_12m_avg_change_percent_tuik"})

    newer_index_only = newer_index[
        newer_index["series_label"].eq("Tüketici Fiyat Endeksi (2025=100)")
        & newer_index["metric_label"].eq("Ölçüm bazında")
    ][["date", "value"]].rename(columns={"value": "cpi_index_2025_100_tuik"})

    newer_mom_only = newer_mom[
        newer_mom["series_label"].str.contains("Aylık Değişim", regex=False)
    ][["date", "value"]].rename(columns={"value": "cpi_mom_percent_tuik_new"})

    monthly = monthly.merge(older_index, on="date", how="left")
    monthly = monthly.merge(older_mom, on="date", how="left")
    monthly = monthly.merge(older_yoy, on="date", how="left")
    monthly = monthly.merge(older_prev_dec, on="date", how="left")
    monthly = monthly.merge(older_rolling, on="date", how="left")
    monthly = monthly.merge(newer_index_only, on="date", how="left")
    monthly = monthly.merge(newer_mom_only, on="date", how="left")

    monthly["cpi_mom_percent_tuik"] = monthly["cpi_mom_percent_tuik"].combine_first(monthly["cpi_mom_percent_tuik_new"])
    monthly["cpi_yoy_percent_tuik_from_2025_base"] = monthly["cpi_index_2025_100_tuik"].pct_change(12) * 100
    monthly["cpi_yoy_percent_tuik"] = monthly["cpi_yoy_percent_tuik"].combine_first(
        monthly["cpi_yoy_percent_tuik_from_2025_base"]
    )

    return monthly[
        [
            "date",
            "cpi_index_1994_100_tuik",
            "cpi_index_2025_100_tuik",
            "cpi_mom_percent_tuik",
            "cpi_yoy_percent_tuik",
            "cpi_12m_avg_change_percent_tuik",
            "cpi_vs_prev_dec_percent_tuik",
        ]
    ].sort_values("date")


def parse_sanayi() -> pd.DataFrame:
    path = find_existing(MANUAL_UPLOADS_DIR / "sanayi_2002_2010.csv", ECONOMY_ROOT / "sanayi_2002_2010.csv")
    frame = parse_tufe_matrix(path)
    output = frame[
        frame["series_label"].eq("Sanayi Üretim Endeksi (2021=100)")
        & frame["metric_label"].str.contains("İmalat", regex=False)
    ][["date", "value"]].rename(columns={"value": "industrial_production_index_tuik"})
    return output.sort_values("date").reset_index(drop=True)


def parse_labor_annual() -> pd.DataFrame:
    unemployment_path = find_existing(MANUAL_UPLOADS_DIR / "ISSIZLIK.csv", ECONOMY_ROOT / "ISSIZLIK.csv")
    employment_path = find_existing(MANUAL_UPLOADS_DIR / "ISTIHDAM.csv", ECONOMY_ROOT / "ISTIHDAM.csv")

    unemployment = pd.read_csv(unemployment_path, encoding="utf-8-sig")
    employment = pd.read_csv(employment_path, encoding="utf-8-sig")

    unemployment["Yıl"] = pd.to_numeric(unemployment["Yıl"], errors="coerce")
    unemployment["İşgücü (Bin Kişi)"] = pd.to_numeric(
        unemployment["İşgücü (Bin Kişi)"].astype(str).str.replace(".", "", regex=False),
        errors="coerce",
    )
    unemployment["İşsiz Sayısı (Bin Kişi)"] = pd.to_numeric(
        unemployment["İşsiz Sayısı (Bin Kişi)"].astype(str).str.replace(".", "", regex=False),
        errors="coerce",
    )
    unemployment["İşsizlik Oranı (%)"] = pd.to_numeric(unemployment["İşsizlik Oranı (%)"], errors="coerce")

    employment["Yıl"] = pd.to_numeric(employment["Yıl"], errors="coerce")
    employment["İstihdam Oranı (%)"] = pd.to_numeric(employment["İstihdam Oranı (%)"], errors="coerce")

    category_map = {
        "1. (15+)": "general_15_plus",
        "2. (15-24)": "youth_15_24",
        "3. (15-64)": "working_age_15_64",
    }

    output = pd.DataFrame({"year": sorted(unemployment["Yıl"].dropna().astype(int).unique())})
    for category, slug in category_map.items():
        subset = unemployment[unemployment["Kategori"] == category].copy()
        subset = subset.rename(
            columns={
                "Yıl": "year",
                "İşgücü (Bin Kişi)": f"labor_force_{slug}_thousand",
                "İşsiz Sayısı (Bin Kişi)": f"unemployed_{slug}_thousand",
                "İşsizlik Oranı (%)": f"unemployment_rate_{slug}_percent",
            }
        )
        output = output.merge(
            subset[
                [
                    "year",
                    f"labor_force_{slug}_thousand",
                    f"unemployed_{slug}_thousand",
                    f"unemployment_rate_{slug}_percent",
                ]
            ],
            on="year",
            how="left",
        )

    employment = employment[employment["Kategori"] == "Genel (15+)"].copy()
    employment = employment.rename(
        columns={
            "Yıl": "year",
            "İstihdam Oranı (%)": "employment_rate_general_15_plus_percent",
        }
    )
    output = output.merge(
        employment[["year", "employment_rate_general_15_plus_percent"]],
        on="year",
        how="left",
    )
    return output.sort_values("year").reset_index(drop=True)


def load_legacy_panel(filename: str, folder: str) -> pd.DataFrame:
    path = find_existing(
        ECONOMY_ROOT / "processed" / folder / filename,
        LEGACY_GENERATED_DIR / "processed_snapshot" / folder / filename,
    )
    return pd.read_csv(path)


def assign_term(period_date: pd.Timestamp) -> int | None:
    for window in TERM_WINDOWS:
        if window["start"] <= period_date <= window["end"]:
            return int(window["term"])
    return None


def add_term_columns(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    output = frame.copy()
    output[date_column] = pd.to_datetime(output[date_column])
    output["term"] = output[date_column].map(assign_term)
    output["term_year"] = output.apply(
        lambda row: int(row[date_column].year) if pd.notna(row["term"]) else pd.NA,
        axis=1,
    )
    output["term_year_key"] = output.apply(
        lambda row: f"{int(row['term'])}_{int(row['term_year'])}"
        if pd.notna(row["term"]) and pd.notna(row["term_year"])
        else pd.NA,
        axis=1,
    )
    return output


def build_daily_panel(fx_daily: pd.DataFrame) -> pd.DataFrame:
    daily = load_legacy_panel("turkey_market_daily_panel.csv", "daily")
    daily["date"] = pd.to_datetime(daily["date"])
    output = daily.merge(fx_daily, on="date", how="left")
    output["usd_try_close_harmonized"] = output["usd_try_close"].combine_first(output["usd_try_tcmb_mid"])
    output["eur_try_close_harmonized"] = output["eur_try_close"].combine_first(output["eur_try_tcmb_mid"])
    return output.sort_values("date").reset_index(drop=True)


def build_monthly_panel(
    fx_monthly: pd.DataFrame,
    policy_monthly: pd.DataFrame,
    reserves_monthly: pd.DataFrame,
    m2_monthly: pd.DataFrame,
    tufe_monthly: pd.DataFrame,
    sanayi_monthly: pd.DataFrame,
) -> pd.DataFrame:
    monthly = load_legacy_panel("turkey_economy_monthly_panel.csv", "monthly")
    monthly["date"] = pd.to_datetime(monthly["date"])

    output = monthly.merge(fx_monthly, on="date", how="left")
    output = output.merge(policy_monthly, on="date", how="left")
    output = output.merge(reserves_monthly, on="date", how="left")
    output = output.merge(m2_monthly, on="date", how="left")
    output = output.merge(tufe_monthly, on="date", how="left")
    output = output.merge(sanayi_monthly, on="date", how="left")

    output["usd_try_avg_harmonized"] = output["usd_try_avg"].combine_first(output["usd_try_tcmb_mid_avg"])
    output["usd_try_last_harmonized"] = output["usd_try_last"].combine_first(output["usd_try_tcmb_mid_last"])
    output["eur_try_avg_harmonized"] = output["eur_try_avg"].combine_first(output["eur_try_tcmb_mid_avg"])
    output["eur_try_last_harmonized"] = output["eur_try_last"].combine_first(output["eur_try_tcmb_mid_last"])
    output["inflation_mom_percent_harmonized"] = output["cpi_mom_percent_tuik"].combine_first(
        output["cpi_inflation_mom_percent_fred"]
    )
    output["inflation_yoy_percent_harmonized"] = output["cpi_yoy_percent_tuik"].combine_first(
        output["cpi_inflation_yoy_percent_fred"]
    )
    output["industrial_production_index_harmonized"] = output["industrial_production_index_tuik"].combine_first(
        output["industrial_production_index"]
    )

    output = add_term_columns(output, date_column="date")
    output["year"] = output["date"].dt.year
    output["month"] = output["date"].dt.month
    return output.sort_values("date").reset_index(drop=True)


def summarize_monthly_to_yearly(monthly: pd.DataFrame) -> pd.DataFrame:
    present_columns = [column for column in MONTH_VALUE_COLUMNS if column in monthly.columns]
    grouped = monthly.groupby("year")
    avg = grouped[present_columns].mean().add_suffix("_annual_avg")
    last = grouped[present_columns].last().add_suffix("_year_end")
    counts = grouped["date"].count().rename("months_with_data")
    summary = avg.join(last).join(counts).reset_index()
    return summary.sort_values("year").reset_index(drop=True)


def build_yearly_panel(monthly_panel: pd.DataFrame, labor_annual: pd.DataFrame) -> pd.DataFrame:
    yearly = load_legacy_panel("turkey_economy_annual_panel.csv", "annual")
    manual_yearly = summarize_monthly_to_yearly(monthly_panel)
    manual_columns = ["year", *[column for column in manual_yearly.columns if column != "year" and column not in yearly.columns]]
    labor_columns = ["year", *[column for column in labor_annual.columns if column != "year" and column not in yearly.columns]]
    output = yearly.merge(manual_yearly[manual_columns], on="year", how="left")
    output = output.merge(labor_annual[labor_columns], on="year", how="left")
    return output.sort_values("year").reset_index(drop=True)


def aggregate_term_groups(monthly_panel: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    data = monthly_panel.dropna(subset=["term"]).copy()
    present_columns = [column for column in MONTH_VALUE_COLUMNS if column in data.columns]
    grouped = data.groupby(group_columns)
    avg = grouped[present_columns].mean().add_suffix("_avg")
    last = grouped[present_columns].last().add_suffix("_last")
    counts = grouped["date"].count().rename("months_in_window")
    start_dates = grouped["date"].min().rename("window_start")
    end_dates = grouped["date"].max().rename("window_end")
    output = avg.join(last).join(counts).join(start_dates).join(end_dates).reset_index()
    return output


def weighted_annual_summary(monthly_panel: pd.DataFrame, yearly_panel: pd.DataFrame) -> pd.DataFrame:
    year_weights = (
        monthly_panel.dropna(subset=["term"])
        .groupby(["term", "year"])
        .size()
        .rename("months_in_year")
        .reset_index()
    )
    annual_columns = [
        column
        for column in yearly_panel.columns
        if column != "year" and pd.api.types.is_numeric_dtype(yearly_panel[column])
    ]
    merged = year_weights.merge(yearly_panel[["year", *annual_columns]], on="year", how="left")
    rows: list[dict[str, float | int]] = []
    for term, group in merged.groupby("term"):
        weights = group["months_in_year"].astype(float)
        row: dict[str, float | int] = {"term": int(term)}
        for column in annual_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & weights.notna()
            if valid.any():
                row[f"annual_weighted_{column}"] = (values[valid] * weights[valid]).sum() / weights[valid].sum()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def build_term_panel(monthly_panel: pd.DataFrame, yearly_panel: pd.DataFrame) -> pd.DataFrame:
    term_panel = aggregate_term_groups(monthly_panel, ["term"])
    annual_summary = weighted_annual_summary(monthly_panel, yearly_panel)
    output = term_panel.merge(annual_summary, on="term", how="left")
    return output.sort_values("term").reset_index(drop=True)


def build_term_year_panel(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    output = aggregate_term_groups(monthly_panel, ["term", "term_year", "term_year_key"])
    return output.sort_values(["term", "term_year"]).reset_index(drop=True)


def build_series_catalog(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for panel_name, frame in frames.items():
        for column in frame.columns:
            if column in {"date", "year", "month", "term", "term_year", "term_year_key", "window_start", "window_end"}:
                continue
            series = frame[column]
            if series.dropna().empty:
                continue
            first_index = series.first_valid_index()
            last_index = series.last_valid_index()
            first_key = None
            last_key = None
            if first_index is not None:
                if "date" in frame.columns:
                    first_key = pd.to_datetime(frame.loc[first_index, "date"]).date().isoformat()
                    last_key = pd.to_datetime(frame.loc[last_index, "date"]).date().isoformat()
                elif "year" in frame.columns:
                    first_key = int(frame.loc[first_index, "year"])
                    last_key = int(frame.loc[last_index, "year"])
                elif "term_year_key" in frame.columns:
                    first_key = frame.loc[first_index, "term_year_key"]
                    last_key = frame.loc[last_index, "term_year_key"]
                elif "term" in frame.columns:
                    first_key = int(frame.loc[first_index, "term"])
                    last_key = int(frame.loc[last_index, "term"])
            records.append(
                {
                    "panel": panel_name,
                    "column_name": column,
                    "non_null_count": int(series.notna().sum()),
                    "first_observation": first_key,
                    "last_observation": last_key,
                }
            )
    return pd.DataFrame(records).sort_values(["panel", "column_name"]).reset_index(drop=True)


def build_term_windows_frame() -> pd.DataFrame:
    return pd.DataFrame(TERM_WINDOWS)[["term", "start", "end"]].rename(columns={"start": "window_start", "end": "window_end"})


def stage_manual_uploads() -> None:
    for filename in [
        "2002_2017_TUFE.csv",
        "2017_2026_AYLIK_TUFE.csv",
        "2017_2026_TUFE_GENEL.csv",
        "ISSIZLIK.csv",
        "ISTIHDAM.csv",
        "dolar_euro.xlsx",
        "gdp_world(including_turkey).xls",
        "kredi_faiz.xlsx",
        "m2_para_arzi.xlsx",
        "mevduat_faiz.xlsx",
        "politika_faizi_1.xlsx",
        "politika_faizi_2.xlsx",
        "sanayi_2002_2010.csv",
        "uluslararasi_rezerv.xlsx",
    ]:
        source = ECONOMY_ROOT / filename
        destination = MANUAL_UPLOADS_DIR / filename
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))


def stage_api_downloads() -> None:
    for folder_name in ["fred_monthly", "market_daily", "world_bank_annual"]:
        source = RAW_DIR / folder_name
        destination = API_DOWNLOADS_DIR / folder_name
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))


def stage_legacy_outputs() -> None:
    processed_source = ECONOMY_ROOT / "processed"
    by_year_source = ECONOMY_ROOT / "by_year"
    if processed_source.exists() and not (LEGACY_GENERATED_DIR / "processed_snapshot").exists():
        shutil.move(str(processed_source), str(LEGACY_GENERATED_DIR / "processed_snapshot"))
    if by_year_source.exists() and not (LEGACY_GENERATED_DIR / "by_year_snapshot").exists():
        shutil.move(str(by_year_source), str(LEGACY_GENERATED_DIR / "by_year_snapshot"))


def remove_obsolete_top_level_entries() -> None:
    for entry in [
        ECONOMY_ROOT / ".DS_Store",
        ECONOMY_ROOT / "notes",
        ECONOMY_ROOT / "scripts",
    ]:
        if entry.is_dir():
            shutil.rmtree(entry)
        elif entry.exists():
            entry.unlink()


def write_outputs(
    daily_panel: pd.DataFrame,
    monthly_panel: pd.DataFrame,
    yearly_panel: pd.DataFrame,
    term_panel: pd.DataFrame,
    term_year_panel: pd.DataFrame,
    series_catalog: pd.DataFrame,
    term_windows: pd.DataFrame,
    kredi_monthly: pd.DataFrame,
    mevduat_monthly: pd.DataFrame,
) -> None:
    daily_panel.to_csv(PREPROCESSED_DIR / "economy_daily_market.csv", index=False, encoding="utf-8-sig")
    monthly_panel.to_csv(PREPROCESSED_DIR / "economy_monthly_macro.csv", index=False, encoding="utf-8-sig")
    yearly_panel.to_csv(PREPROCESSED_DIR / "economy_yearly_network_panel.csv", index=False, encoding="utf-8-sig")
    term_panel.to_csv(PREPROCESSED_DIR / "economy_term_network_panel.csv", index=False, encoding="utf-8-sig")
    term_year_panel.to_csv(PREPROCESSED_DIR / "economy_term_year_network_panel.csv", index=False, encoding="utf-8-sig")
    series_catalog.to_csv(PREPROCESSED_DIR / "economy_series_catalog.csv", index=False, encoding="utf-8-sig")
    term_windows.to_csv(PREPROCESSED_DIR / "economy_term_windows.csv", index=False, encoding="utf-8-sig")
    kredi_monthly.to_csv(PREPROCESSED_DIR / "economy_kredi_faiz_monthly_codes.csv", index=False, encoding="utf-8-sig")
    mevduat_monthly.to_csv(PREPROCESSED_DIR / "economy_mevduat_faiz_monthly_codes.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()

    fx_daily, fx_monthly = parse_manual_fx()
    policy_monthly = parse_policy_rate()
    reserves_monthly = parse_reserves()
    m2_monthly = parse_m2()
    tufe_monthly = parse_tufe()
    sanayi_monthly = parse_sanayi()
    labor_annual = parse_labor_annual()
    kredi_monthly = parse_weekly_evds_file("kredi_faiz.xlsx")
    mevduat_monthly = parse_weekly_evds_file("mevduat_faiz.xlsx")

    daily_panel = build_daily_panel(fx_daily)
    monthly_panel = build_monthly_panel(
        fx_monthly=fx_monthly,
        policy_monthly=policy_monthly,
        reserves_monthly=reserves_monthly,
        m2_monthly=m2_monthly,
        tufe_monthly=tufe_monthly,
        sanayi_monthly=sanayi_monthly,
    )
    yearly_panel = build_yearly_panel(monthly_panel=monthly_panel, labor_annual=labor_annual)
    term_panel = build_term_panel(monthly_panel=monthly_panel, yearly_panel=yearly_panel)
    term_year_panel = build_term_year_panel(monthly_panel=monthly_panel)
    series_catalog = build_series_catalog(
        {
            "economy_daily_market": daily_panel,
            "economy_monthly_macro": monthly_panel,
            "economy_yearly_network_panel": yearly_panel,
            "economy_term_network_panel": term_panel,
            "economy_term_year_network_panel": term_year_panel,
        }
    )
    term_windows = build_term_windows_frame()

    write_outputs(
        daily_panel=daily_panel,
        monthly_panel=monthly_panel,
        yearly_panel=yearly_panel,
        term_panel=term_panel,
        term_year_panel=term_year_panel,
        series_catalog=series_catalog,
        term_windows=term_windows,
        kredi_monthly=kredi_monthly,
        mevduat_monthly=mevduat_monthly,
    )

    stage_manual_uploads()
    stage_api_downloads()
    stage_legacy_outputs()
    remove_obsolete_top_level_entries()


if __name__ == "__main__":
    main()
