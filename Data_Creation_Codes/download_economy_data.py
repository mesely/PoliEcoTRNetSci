from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
ECONOMY_ROOT = SCRIPT_DIR.parent
RAW_DIR = ECONOMY_ROOT / "raw"
PROCESSED_DIR = ECONOMY_ROOT / "processed"
BY_YEAR_DIR = ECONOMY_ROOT / "by_year"
NOTES_DIR = ECONOMY_ROOT / "notes"

START_DATE = "2002-01-01"
DEFAULT_END_DATE = date.today().isoformat()

YAHOO_SERIES = {
    "usd_try": {
        "symbol": "USDTRY=X",
        "label": "USD/TRY daily exchange rate",
        "close_column": "usd_try_close",
        "source": "Yahoo Finance chart endpoint",
    },
    "eur_try": {
        "symbol": "EURTRY=X",
        "label": "EUR/TRY daily exchange rate",
        "close_column": "eur_try_close",
        "source": "Yahoo Finance chart endpoint",
    },
    "bist100": {
        "symbol": "XU100.IS",
        "label": "BIST 100 daily index",
        "close_column": "bist100_close",
        "source": "Yahoo Finance chart endpoint",
    },
    "gold_usd": {
        "symbol": "GC=F",
        "label": "Gold futures daily price, USD",
        "close_column": "gold_usd_close",
        "source": "Yahoo Finance chart endpoint",
    },
    "brent_oil_usd": {
        "symbol": "BZ=F",
        "label": "Brent crude oil futures daily price, USD",
        "close_column": "brent_oil_usd_close",
        "source": "Yahoo Finance chart endpoint",
    },
    "wti_oil_usd": {
        "symbol": "CL=F",
        "label": "WTI crude oil futures daily price, USD",
        "close_column": "wti_oil_usd_close",
        "source": "Yahoo Finance chart endpoint",
    },
}

WORLD_BANK_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_annual_percent",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_current_usd",
    "FP.CPI.TOTL.ZG": "inflation_cpi_annual_percent",
    "FP.CPI.TOTL": "cpi_index_2010_100",
    "SL.UEM.TOTL.ZS": "unemployment_total_percent",
    "BN.CAB.XOKA.GD.ZS": "current_account_balance_gdp_percent",
    "PA.NUS.FCRF": "official_exchange_rate_lcu_per_usd",
    "FR.INR.RINR": "real_interest_rate_percent",
    "FR.INR.LEND": "lending_interest_rate_percent",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_net_inflows_gdp_percent",
    "NY.GDP.PCAP.KD.ZG": "gdp_per_capita_growth_percent",
    "NE.TRD.GNFS.ZS": "trade_gdp_percent",
    "NE.EXP.GNFS.ZS": "exports_gdp_percent",
    "NE.IMP.GNFS.ZS": "imports_gdp_percent",
    "NE.RSB.GNFS.ZS": "external_balance_goods_services_gdp_percent",
    "DT.DOD.DECT.GN.ZS": "external_debt_gni_percent",
    "FI.RES.TOTL.CD": "total_reserves_usd",
    "GC.TAX.TOTL.GD.ZS": "tax_revenue_gdp_percent",
    "GC.REV.XGRT.GD.ZS": "revenue_excluding_grants_gdp_percent",
    "GC.XPN.TOTL.GD.ZS": "expense_gdp_percent",
    "GC.DOD.TOTL.GD.ZS": "central_government_debt_gdp_percent",
    "GC.NLD.TOTL.GD.ZS": "net_lending_borrowing_gdp_percent",
    "NY.GNS.ICTR.ZS": "gross_savings_gdp_percent",
    "NE.GDI.TOTL.ZS": "gross_capital_formation_gdp_percent",
    "NE.CON.TOTL.ZS": "final_consumption_expenditure_gdp_percent",
    "NE.CON.GOVT.ZS": "government_consumption_gdp_percent",
    "NV.IND.TOTL.ZS": "industry_value_added_gdp_percent",
    "NV.SRV.TOTL.ZS": "services_value_added_gdp_percent",
    "NV.AGR.TOTL.ZS": "agriculture_value_added_gdp_percent",
    "SI.POV.GINI": "gini_index",
    "SP.POP.TOTL": "population_total",
    "SP.POP.GROW": "population_growth_percent",
    "SL.TLF.CACT.ZS": "labor_force_participation_percent",
    "SL.EMP.TOTL.SP.ZS": "employment_to_population_percent",
    "FB.AST.NPER.ZS": "bank_nonperforming_loans_percent",
    "FS.AST.PRVT.GD.ZS": "domestic_credit_private_sector_gdp_percent",
    "FM.LBL.BMNY.GD.ZS": "broad_money_gdp_percent",
}

FRED_SERIES = {
    "IR3TIB01TRM156N": "short_term_interest_rate_percent",
    "INTDSRTRM193N": "discount_rate_percent",
    "IRSTCI01TRM156N": "immediate_interbank_rate_percent",
    "TURCPIALLMINMEI": "cpi_all_items_index_fred",
    "CPALTT01TRM659N": "cpi_inflation_yoy_percent_fred",
    "CPALTT01TRM657N": "cpi_inflation_mom_percent_fred",
    "LRHUTTTTTRM156S": "harmonized_unemployment_rate_percent",
    "TURPROINDMISMEI": "industrial_production_index",
    "XTEXVA01TRM664S": "exports_value_fred",
    "XTIMVA01TRM664S": "imports_value_fred",
}


def ensure_dirs() -> None:
    for path in [
        RAW_DIR / "market_daily",
        RAW_DIR / "world_bank_annual",
        RAW_DIR / "fred_monthly",
        PROCESSED_DIR / "daily",
        PROCESSED_DIR / "monthly",
        PROCESSED_DIR / "annual",
        BY_YEAR_DIR,
        NOTES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def unix_timestamp(day: str) -> int:
    parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        url,
        params=params,
        timeout=40,
        headers={"User-Agent": "Mozilla/5.0 PoliEcoTRNetworSci/1.0"},
    )
    response.raise_for_status()
    return response.json()


def request_text(url: str, params: dict[str, Any] | None = None) -> str:
    response = requests.get(
        url,
        params=params,
        timeout=40,
        headers={"User-Agent": "Mozilla/5.0 PoliEcoTRNetworSci/1.0"},
    )
    response.raise_for_status()
    return response.text


def download_yahoo_daily(key: str, start_date: str, end_date: str) -> pd.DataFrame:
    meta = YAHOO_SERIES[key]
    params = {
        "period1": unix_timestamp(start_date),
        "period2": unix_timestamp(end_date) + 24 * 60 * 60,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{meta['symbol']}"
    payload = request_json(url, params=params)
    (RAW_DIR / "market_daily" / f"{key}_yahoo_chart_raw.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo returned no rows for {key}")
    result0 = result[0]
    timestamps = result0.get("timestamp", [])
    quote = result0.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result0.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
    rows: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                "open": quote.get("open", [None] * len(timestamps))[idx],
                "high": quote.get("high", [None] * len(timestamps))[idx],
                "low": quote.get("low", [None] * len(timestamps))[idx],
                "close": quote.get("close", [None] * len(timestamps))[idx],
                "adjclose": adjclose[idx] if idx < len(adjclose) else None,
                "volume": quote.get("volume", [None] * len(timestamps))[idx],
                "symbol": meta["symbol"],
                "series": key,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df.to_csv(RAW_DIR / "market_daily" / f"{key}_daily.csv", index=False, encoding="utf-8-sig")
    return df


def parse_tcmb_rate(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    numeric = float(str(value).replace(",", "."))
    if numeric > 1000:
        numeric = numeric / 1_000_000
    return numeric


def fetch_tcmb_usd_for_day(day: pd.Timestamp) -> dict[str, Any] | None:
    day_code = day.strftime("%d%m%Y")
    url = f"https://www.tcmb.gov.tr/kurlar/{day:%Y%m}/{day_code}.xml"
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 PoliEcoTRNetworSci/1.0"},
        )
    except requests.RequestException:
        return None
    if response.status_code != 200 or not response.content.strip().startswith(b"<?xml"):
        return None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None
    for currency in root.findall("Currency"):
        if currency.attrib.get("CurrencyCode") == "USD" or currency.attrib.get("Kod") == "USD":
            forex_buying = parse_tcmb_rate(currency.findtext("ForexBuying"))
            forex_selling = parse_tcmb_rate(currency.findtext("ForexSelling"))
            if forex_buying is None and forex_selling is None:
                return None
            close = (
                (forex_buying + forex_selling) / 2
                if forex_buying is not None and forex_selling is not None
                else forex_buying or forex_selling
            )
            return {
                "date": day.date().isoformat(),
                "open": close,
                "high": max(value for value in [forex_buying, forex_selling] if value is not None),
                "low": min(value for value in [forex_buying, forex_selling] if value is not None),
                "close": close,
                "adjclose": close,
                "volume": None,
                "symbol": "TCMB_USDTRY",
                "series": "usd_try",
                "forex_buying": forex_buying,
                "forex_selling": forex_selling,
                "source": "TCMB daily exchange-rate XML archive",
            }
    return None


def download_tcmb_usd_try_supplement(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end) or end < start:
        return pd.DataFrame()
    business_days = list(pd.bdate_range(start, end))
    if not business_days:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for row in executor.map(fetch_tcmb_usd_for_day, business_days):
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(
        RAW_DIR / "market_daily" / "usd_try_tcmb_daily_supplement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df


def combine_usd_try_sources(yahoo_usd_try: pd.DataFrame, start_date: str) -> pd.DataFrame:
    if yahoo_usd_try.empty:
        return yahoo_usd_try
    first_yahoo_date = yahoo_usd_try["date"].min()
    supplement_end = (first_yahoo_date - pd.Timedelta(days=1)).date().isoformat()
    supplement = download_tcmb_usd_try_supplement(start_date, supplement_end)
    frames = [supplement, yahoo_usd_try] if not supplement.empty else [yahoo_usd_try]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    combined.to_csv(
        RAW_DIR / "market_daily" / "usd_try_combined_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return combined


def download_world_bank() -> pd.DataFrame:
    long_frames: list[pd.DataFrame] = []
    for indicator, column_name in WORLD_BANK_INDICATORS.items():
        payload = request_json(
            f"https://api.worldbank.org/v2/country/TUR/indicator/{indicator}",
            params={"format": "json", "per_page": 20000},
        )
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        raw_path = RAW_DIR / "world_bank_annual" / f"{indicator}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame = frame[["date", "value"]].copy()
        frame["year"] = pd.to_numeric(frame["date"], errors="coerce").astype("Int64")
        frame["indicator_code"] = indicator
        frame["indicator_name"] = column_name
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        long_frames.append(frame[["year", "indicator_code", "indicator_name", "value"]])
        time.sleep(0.1)

    if not long_frames:
        return pd.DataFrame(columns=["year"])

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df[long_df["year"].between(2002, date.today().year)]
    long_df.to_csv(
        RAW_DIR / "world_bank_annual" / "world_bank_turkey_macro_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wide_df = (
        long_df.pivot_table(index="year", columns="indicator_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values("year")
    )
    wide_df.to_csv(
        RAW_DIR / "world_bank_annual" / "world_bank_turkey_macro_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return wide_df


def download_fred_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series_id, column_name in FRED_SERIES.items():
        text = request_text("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
        raw_path = RAW_DIR / "fred_monthly" / f"{series_id}.csv"
        raw_path.write_text(text, encoding="utf-8")
        frame = pd.read_csv(io.StringIO(text))
        if frame.empty:
            continue
        value_col = [col for col in frame.columns if col != "observation_date"][0]
        frame = frame.rename(columns={"observation_date": "date", value_col: column_name})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
        frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
        frames.append(frame[["date", column_name]])
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame(columns=["date"])
    monthly = frames[0]
    for frame in frames[1:]:
        monthly = monthly.merge(frame, on="date", how="outer")
    monthly = monthly.sort_values("date").reset_index(drop=True)
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly.to_csv(
        RAW_DIR / "fred_monthly" / "fred_turkey_interest_monthly_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return monthly


def return_pct(values: pd.Series) -> float | None:
    clean = values.dropna()
    if clean.size < 2 or clean.iloc[0] == 0:
        return None
    return ((clean.iloc[-1] / clean.iloc[0]) - 1) * 100


def close_columns(daily_panel: pd.DataFrame) -> list[str]:
    return [column for column in daily_panel.columns if column.endswith("_close")]


def build_daily_market_panel(market_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel: pd.DataFrame | None = None
    for key, frame in market_frames.items():
        close_column = str(YAHOO_SERIES[key]["close_column"])
        source = frame[["date", "close"]].rename(columns={"close": close_column})
        panel = source if panel is None else panel.merge(source, on="date", how="outer")
    if panel is None:
        panel = pd.DataFrame(columns=["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    for column in close_columns(panel):
        panel[f"{column.removesuffix('_close')}_return_1d_pct"] = panel[column].pct_change(fill_method=None) * 100
    panel.to_csv(
        PROCESSED_DIR / "daily" / "turkey_market_daily_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return panel


def build_monthly_panel(daily_panel: pd.DataFrame, fred_monthly: pd.DataFrame) -> pd.DataFrame:
    daily = daily_panel.copy()
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    records: list[dict[str, Any]] = []
    for month_start, block in daily.groupby("month_start", dropna=False):
        record: dict[str, Any] = {"date": month_start, "market_days": block["date"].nunique()}
        for column in close_columns(block):
            prefix = column.removesuffix("_close")
            record[f"{prefix}_avg"] = block[column].mean()
            record[f"{prefix}_last"] = block[column].dropna().iloc[-1] if block[column].dropna().size else None
            record[f"{prefix}_return_month_pct"] = return_pct(block[column])
        records.append(record)
    monthly_market = pd.DataFrame(records)
    monthly_market["year"] = monthly_market["date"].dt.year
    monthly_market["month"] = monthly_market["date"].dt.month
    monthly = monthly_market.merge(fred_monthly, on=["date", "year", "month"], how="outer")
    monthly = monthly.sort_values("date").reset_index(drop=True)
    monthly.to_csv(
        PROCESSED_DIR / "monthly" / "turkey_economy_monthly_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return monthly


def build_annual_panel(daily_panel: pd.DataFrame, monthly_panel: pd.DataFrame, world_bank: pd.DataFrame) -> pd.DataFrame:
    market_records: list[dict[str, Any]] = []
    for year, block in daily_panel.groupby("year", dropna=False):
        record: dict[str, Any] = {"year": year, "market_days": block["date"].nunique()}
        for column in close_columns(block):
            prefix = column.removesuffix("_close")
            record[f"{prefix}_avg"] = block[column].mean()
            record[f"{prefix}_last"] = block[column].dropna().iloc[-1] if block[column].dropna().size else None
            record[f"{prefix}_return_year_pct"] = return_pct(block[column])
        market_records.append(record)
    annual_market = pd.DataFrame(market_records)

    monthly_numeric = [
        column
        for column in monthly_panel.columns
        if column not in {"date", "year", "month"} and pd.api.types.is_numeric_dtype(monthly_panel[column])
    ]
    annual_monthly = (
        monthly_panel.groupby("year", as_index=False)[monthly_numeric].mean()
        if monthly_numeric
        else pd.DataFrame(columns=["year"])
    )
    annual_monthly = annual_monthly.rename(columns={column: f"{column}_annual_avg" for column in monthly_numeric})
    annual = annual_market.merge(annual_monthly, on="year", how="outer")
    if not world_bank.empty:
        annual = annual.merge(world_bank, on="year", how="outer")
    annual = annual[(annual["year"] >= 2002) & (annual["year"] <= date.today().year)]
    annual = annual.sort_values("year").reset_index(drop=True)
    annual.to_csv(
        PROCESSED_DIR / "annual" / "turkey_economy_annual_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return annual


def write_yearly_slices(daily_panel: pd.DataFrame, monthly_panel: pd.DataFrame, annual_panel: pd.DataFrame) -> None:
    for year in range(2002, date.today().year + 1):
        year_dir = BY_YEAR_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        daily_panel[daily_panel["year"] == year].to_csv(
            year_dir / f"economy_{year}_daily_market.csv",
            index=False,
            encoding="utf-8-sig",
        )
        monthly_panel[monthly_panel["year"] == year].to_csv(
            year_dir / f"economy_{year}_monthly_market_interest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        annual_panel[annual_panel["year"] == year].to_csv(
            year_dir / f"economy_{year}_annual_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


def write_notes(start_date: str, end_date: str, annual_panel: pd.DataFrame) -> None:
    source_lines = [
        "# Economy Data Sources",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Date window: {start_date} to {end_date}",
        "",
        "## Market daily series",
    ]
    for key, meta in YAHOO_SERIES.items():
        source_lines.append(f"- {meta['label']}: Yahoo Finance chart endpoint, symbol `{meta['symbol']}`.")
    source_lines.extend(
        [
        "- USD/TRY supplement before Yahoo coverage starts: TCMB daily exchange-rate XML archive.",
        "- TCMB pre-2005 old-TL USD rates are divided by 1,000,000 to align them with the new TRY scale.",
        "",
        "## Annual macro series",
        "- World Bank API, country `TUR`, indicators:",
        ]
    )
    for code, name in WORLD_BANK_INDICATORS.items():
        source_lines.append(f"- `{code}` -> `{name}`")
    source_lines.extend(
        [
            "",
            "## Monthly/high-frequency macro series",
            "- FRED CSV endpoint, OECD/FRED macro series ids:",
        ]
    )
    for code, name in FRED_SERIES.items():
        source_lines.append(f"- `{code}` -> `{name}`")
    source_lines.extend(
        [
            "",
            "## Use in project",
            "- Use `processed/annual/turkey_economy_annual_panel.csv` for term/event-level annual joins.",
            "- Use `processed/monthly/turkey_economy_monthly_panel.csv` for finer timing around major events.",
            "- Use `by_year/<year>/` folders when comparing yearly network outputs with market/macroeconomic movement.",
        ]
    )
    (NOTES_DIR / "DATA_SOURCES.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# economy_data",
        "",
        "This folder stores reproducible Turkish macro/market data for the TBMM network project.",
        "",
        "## Folder map",
        "- `raw/market_daily`: source-level Yahoo chart outputs and normalized daily CSVs.",
        "- `raw/world_bank_annual`: World Bank raw JSON plus long/wide annual macro tables.",
        "- `raw/fred_monthly`: FRED raw monthly interest-rate CSVs.",
        "- `processed/daily`: combined market panel: USD/TRY, EUR/TRY, BIST 100, gold, Brent oil, WTI oil.",
        "- `processed/monthly`: monthly market, inflation, labor, industry, trade, and interest-rate panel.",
        "- `processed/annual`: annual panel for event and term-level joins.",
        "- `by_year`: one folder per year with daily, monthly, and annual slices.",
        "- `scripts`: downloader/processor script.",
        "- `notes`: source and run notes.",
        "",
        "## Annual coverage snapshot",
        f"- First year: {int(annual_panel['year'].min()) if not annual_panel.empty else 'NA'}",
        f"- Last year: {int(annual_panel['year'].max()) if not annual_panel.empty else 'NA'}",
        f"- Rows: {len(annual_panel)}",
    ]
    (NOTES_DIR / "README.md").write_text("\n".join(map(str, readme_lines)) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and organize Turkish economy data for network analysis.")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    print(f"Downloading market daily data: {args.start_date} to {args.end_date}")
    market_frames: dict[str, pd.DataFrame] = {}
    for key in YAHOO_SERIES:
        frame = download_yahoo_daily(key, args.start_date, args.end_date)
        if key == "usd_try":
            frame = combine_usd_try_sources(frame, args.start_date)
        market_frames[key] = frame

    print("Downloading World Bank annual macro data")
    world_bank = download_world_bank()

    print("Downloading FRED monthly interest data")
    fred_monthly = download_fred_monthly(args.start_date, args.end_date)

    print("Building processed panels")
    daily_panel = build_daily_market_panel(market_frames)
    monthly_panel = build_monthly_panel(daily_panel, fred_monthly)
    annual_panel = build_annual_panel(daily_panel, monthly_panel, world_bank)
    write_yearly_slices(daily_panel, monthly_panel, annual_panel)
    write_notes(args.start_date, args.end_date, annual_panel)

    print(f"Done. economy_data written to: {ECONOMY_ROOT}")
    print(f"Annual panel rows: {len(annual_panel)}")
    print(f"Daily panel rows: {len(daily_panel)}")
    print(f"Monthly panel rows: {len(monthly_panel)}")


if __name__ == "__main__":
    main()
