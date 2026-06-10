"""
build_multiterm_analyses.py
────────────────────────────────────────────────────────────────────────────
Generate comprehensive + institutional-conflict figures for Terms 23–28.
Each term produces two publication-ready PNGs (except Term 25 which is
simplified due to 120-speech sample).

Outputs per term
  term_XX_comprehensive_analysis.png  – 6-panel overview
  term_XX_institutional_conflict.png  – 4-panel democratic-stress analysis
Archives
  term_XX_total_yearly_networks.png   → Term_XX/Figures/archive/
"""

import os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import ttest_ind, pearsonr

warnings.filterwarnings('ignore')

# ── colour palette ─────────────────────────────────────────────────────────
PARTY_COLORS = {
    'Adalet ve Kalkınma Partisi':           '#E63329',   # AKP – red
    'Cumhuriyet Halk Partisi':              '#E87722',   # CHP – orange/kemalist
    'Milliyetçi Hareket Partisi':           '#8B0000',   # MHP – dark red
    'Halkların Demokratik Partisi':         '#6A0DAD',   # HDP – purple
    'Halkların Eşitlik ve Demokrasi Partisi': '#6A0DAD', # HEDEP – purple
    'İYİ Parti':                            '#00A0E3',   # IYI – blue
    'YENİ YOL Partisi':                     '#2CA02C',   # Yeni Yol – green
    'Demokratik Sol Parti':                 '#1F77B4',   # DSP – blue
    'Bağımsız':                             '#888888',
}
GOV_PARTY = 'Adalet ve Kalkınma Partisi'   # AKP governs all terms 23-28

# ── term metadata ───────────────────────────────────────────────────────────
TERM_META = {
    23: {
        'label': 'Term 23 (2007–2011)',
        'years': [2007, 2008, 2009, 2010, 2011],
        'events': [
            (2008.3, 'AKP Kapatma\nDavası'),
            (2008.7, 'Küresel Kriz\nBaşlangıcı'),
            (2010.8, 'Anayasa\nReferandum'),
        ],
    },
    24: {
        'label': 'Term 24 (2011–2015)',
        'years': [2011, 2012, 2013, 2014, 2015],
        'events': [
            (2013.4, 'Gezi Parkı\nProtestolar'),
            (2013.95, '17-25 Aralık\nYolsuzluk'),
        ],
    },
    25: {
        'label': 'Term 25 (2015 Geçiş)',
        'years': [2015],
        'events': [],
    },
    26: {
        'label': 'Term 26 (2015–2018)',
        'years': [2015, 2016, 2017, 2018],
        'events': [
            (2016.55, '15 Temmuz\nDarbe Girişimi'),
            (2017.3,  'Anayasa\nReferandum'),
        ],
    },
    27: {
        'label': 'Term 27 (2018–2023)',
        'years': [2018, 2019, 2020, 2021, 2022, 2023],
        'events': [
            (2020.2,  'COVID-19'),
            (2022.08, 'Ekonomik\nKriz'),
            (2023.4,  '2023\nSeçimleri'),
        ],
    },
    28: {
        'label': 'Term 28 (2023–)',
        'years': [2023, 2024, 2025],
        'events': [
            (2024.2, 'Yerel\nSeçimler'),
        ],
    },
}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in src/)


# ── helpers ─────────────────────────────────────────────────────────────────

def load_csvs(term):
    p = f'{BASE}/Term_{term}/CSVs/term_{term}_'
    return {
        'speech':    pd.read_csv(p + 'beme_speech_labels.csv'),
        'concept':   pd.read_csv(p + 'beme_concept_edges.csv'),
        'y_party':   pd.read_csv(p + 'yearly_layer_party_concept_edges.csv'),
        'y_net':     pd.read_csv(p + 'yearly_layer_network_summary.csv'),
        'l_party':   pd.read_csv(p + 'layer_party_metrics.csv'),
    }


def gini_lorenz(values):
    """Return (gini, lorenz_x, lorenz_y)."""
    v = np.sort(values)
    n = len(v)
    cumv = np.cumsum(v)
    lorenz_y = np.concatenate([[0], cumv / cumv[-1]])
    lorenz_x = np.linspace(0, 1, n + 1)
    area = np.trapezoid(lorenz_y, lorenz_x)
    gini = 1 - 2 * area
    return gini, lorenz_x, lorenz_y


def powerlaw_fit(values):
    """Log-log OLS for rank-frequency.  Returns (alpha, r2)."""
    counts = np.sort(values)[::-1]
    ranks  = np.arange(1, len(counts) + 1)
    mask   = counts > 0
    log_r  = np.log(ranks[mask])
    log_c  = np.log(counts[mask])
    slope, intercept, r, *_ = stats.linregress(log_r, log_c)
    return abs(slope), r**2


def mann_kendall(x):
    """Return (Z, p) for Mann-Kendall trend test."""
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    s = sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i+1, n))
    v = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(v) if v > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def topic_yearly(y_party, years):
    """Return df with yearly share of each concept_category mention for AKP."""
    df = y_party[y_party['layer'] == 'total'].copy()
    cats = ['macro_economy', 'constitutional_conflict', 'democratic_reform', 'security_conflict']
    rows = []
    for yr in years:
        ydf = df[df['year'] == yr]
        total = ydf['mention_count'].sum()
        if total == 0:
            continue
        row = {'year': yr}
        for c in cats:
            row[c] = ydf[ydf['concept_category'] == c]['mention_count'].sum() / total
        rows.append(row)
    return pd.DataFrame(rows).set_index('year')


def cc_by_party_yearly(concept, years):
    """cc_neg and dr_pos and eco_neg yearly by party."""
    gov_mask  = concept['party'] == GOV_PARTY
    opp_mask  = concept['party'] != GOV_PARTY
    out = {}
    for yr in years:
        y = concept[concept['year'] == yr]
        cc  = y[y['concept_category'] == 'constitutional_conflict']
        dr  = y[y['concept_category'] == 'democratic_reform']
        eco = y[y['concept_category'] == 'macro_economy']
        out[yr] = {
            'cc_neg_gov':  cc[cc['party'] == GOV_PARTY]['beme_negative_hits'].sum(),
            'cc_neg_opp':  cc[opp_mask & (y['concept_category'] == 'constitutional_conflict')]['beme_negative_hits'].sum(),
            'cc_men_gov':  cc[cc['party'] == GOV_PARTY]['mention_count'].sum(),
            'cc_men_opp':  cc[opp_mask & (y['concept_category'] == 'constitutional_conflict')]['mention_count'].sum(),
            'dr_pos_gov':  dr[dr['party'] == GOV_PARTY]['beme_positive_hits'].sum(),
            'dr_pos_opp':  dr[opp_mask & (y['concept_category'] == 'democratic_reform')]['beme_positive_hits'].sum(),
            'eco_neg_gov': eco[eco['party'] == GOV_PARTY]['beme_negative_hits'].sum(),
            'eco_neg_opp': eco[opp_mask & (y['concept_category'] == 'macro_economy')]['beme_negative_hits'].sum(),
        }
    return pd.DataFrame(out).T


def demo_stress_index(df_stats):
    """Democratic stress = (cc_neg_total) / (dr_pos_total + 0.01)."""
    cc = df_stats['cc_neg_gov'] + df_stats['cc_neg_opp']
    dr = df_stats['dr_pos_gov'] + df_stats['dr_pos_opp']
    return cc / (dr + 0.01)


# ── figure 1: comprehensive analysis ────────────────────────────────────────

def plot_comprehensive(term, data, meta, outdir):
    speech  = data['speech']
    concept = data['concept']
    y_party = data['y_party']
    y_net   = data['y_net']
    years   = meta['years']
    label   = meta['label']
    events  = meta['events']

    simplified = (len(speech) < 200)   # Term 25

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(f'PoliEcoTR — {label}: Comprehensive Network Analysis',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    # ── A: Speaker power law / Lorenz ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    sp_counts = speech.groupby('speaker').size().values
    gini, lx, ly = gini_lorenz(sp_counts)
    if len(sp_counts) >= 10 and not simplified:
        alpha, r2 = powerlaw_fit(sp_counts)
        ax_a.fill_between(lx, ly, lx, alpha=0.25, color='#2196F3')
        ax_a.plot(lx, ly, color='#2196F3', lw=2, label=f'Lorenz (Gini={gini:.3f})')
        ax_a.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Equality')
        ax_a.set_xlabel('Cumulative Speaker Share', fontsize=9)
        ax_a.set_ylabel('Cumulative Speech Share', fontsize=9)
        ax_a.set_title(f'A. Speaker Inequality\nGini={gini:.3f}, α={alpha:.2f}', fontsize=10, fontweight='bold')
        ax_a.legend(fontsize=8)
    else:
        # Simplified bar chart for small samples
        sp_top = speech.groupby('speaker').size().sort_values(ascending=False).head(10)
        clrs   = [PARTY_COLORS.get(
                    speech[speech['speaker'] == s]['party'].iloc[0], '#999')
                  for s in sp_top.index]
        ax_a.barh(range(len(sp_top)), sp_top.values, color=clrs)
        ax_a.set_yticks(range(len(sp_top)))
        ax_a.set_yticklabels([s[:20] for s in sp_top.index], fontsize=7)
        ax_a.set_xlabel('Speech Count', fontsize=9)
        ax_a.set_title(f'A. Top Speakers\n(N={len(speech)} speeches)', fontsize=10, fontweight='bold')

    ax_a.set_facecolor('#F5F5F5')
    ax_a.spines[['top', 'right']].set_visible(False)

    # ── B: Topic evolution (stacked area) ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cats  = ['macro_economy', 'constitutional_conflict', 'democratic_reform', 'security_conflict']
    cmap  = ['#E63329', '#8B1A1A', '#2196F3', '#888888']
    clabels = ['Macro Economy', 'Constitutional Conflict', 'Democratic Reform', 'Security']

    top_df = topic_yearly(y_party, years)
    if len(top_df) > 1:
        xs = top_df.index.astype(float)
        bottom = np.zeros(len(top_df))
        for c, col, lbl in zip(cats, cmap, clabels):
            vals = top_df.get(c, pd.Series(0, index=top_df.index)).fillna(0).values
            ax_b.bar(xs, vals, bottom=bottom, color=col, alpha=0.82,
                     label=lbl, width=0.65)
            bottom += vals
        ax_b.set_xlim(min(xs)-0.5, max(xs)+0.5)
        ax_b.set_xticks(xs)
        ax_b.set_xticklabels([str(int(y)) for y in xs], fontsize=8)
        ax_b.set_ylabel('Share of Mentions', fontsize=9)
        ax_b.legend(fontsize=7, loc='upper right')
    else:
        # Term 25: single bar
        ax_b.text(0.5, 0.5, 'Single-year term\n(2015 Geçiş)',
                  ha='center', va='center', transform=ax_b.transAxes, fontsize=11)

    ax_b.set_title('B. Topic Evolution by Year', fontsize=10, fontweight='bold')
    ax_b.set_facecolor('#F5F5F5')
    ax_b.spines[['top', 'right']].set_visible(False)
    for ev_yr, ev_lbl in events:
        ax_b.axvline(ev_yr, color='#333', lw=1.2, ls='--', alpha=0.6)
        ax_b.text(ev_yr, 1.02, ev_lbl, ha='center', va='bottom',
                  fontsize=6.5, transform=ax_b.get_xaxis_transform(), color='#333')

    # ── C: Network density & assortativity ──────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ynet = y_net[y_net['layer'] == 'total'].sort_values('year')
    if len(ynet) > 1:
        ax_c2 = ax_c.twinx()
        ax_c.plot(ynet['year'], ynet['party_projection_density'],
                  'o-', color='#2196F3', lw=2, ms=5, label='Party Density')
        ax_c2.plot(ynet['year'], ynet['concept_projection_density'],
                   's--', color='#E63329', lw=2, ms=5, label='Concept Density')
        ax_c.set_ylabel('Party Proj. Density', color='#2196F3', fontsize=8)
        ax_c2.set_ylabel('Concept Proj. Density', color='#E63329', fontsize=8)
        ax_c.set_xlabel('Year', fontsize=9)
        # combined legend
        lines1, labs1 = ax_c.get_legend_handles_labels()
        lines2, labs2 = ax_c2.get_legend_handles_labels()
        ax_c.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')
    else:
        ax_c.text(0.5, 0.5, 'Single-year\nnetwork', ha='center', va='center',
                  transform=ax_c.transAxes, fontsize=11)

    ax_c.set_title('C. Network Density Over Time', fontsize=10, fontweight='bold')
    ax_c.set_facecolor('#F5F5F5')
    ax_c.spines[['top', 'right']].set_visible(False)

    # ── D: Party BEME comparison (pos vs neg per party) ─────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    party_stats = (concept.groupby('party')
                          .agg(pos=('beme_positive_hits', 'sum'),
                               neg=('beme_negative_hits', 'sum'))
                          .nlargest(5, 'pos'))
    party_stats['pos_share'] = party_stats['pos'] / (party_stats['pos'] + party_stats['neg'] + 0.001)
    party_stats['neg_share'] = party_stats['neg'] / (party_stats['pos'] + party_stats['neg'] + 0.001)

    pnames = [p[:18] for p in party_stats.index]
    xp = range(len(party_stats))
    ax_d.barh(xp, party_stats['pos_share'], color='#2196F3', alpha=0.8, label='Positive')
    ax_d.barh(xp, -party_stats['neg_share'], color='#E63329', alpha=0.8, label='Negative')
    ax_d.set_yticks(list(xp))
    ax_d.set_yticklabels(pnames, fontsize=7.5)
    ax_d.axvline(0, color='black', lw=0.8)
    ax_d.set_xlabel('← Negative Share | Positive Share →', fontsize=8)
    ax_d.legend(fontsize=8)
    ax_d.set_title('D. Party BEME Sentiment Balance', fontsize=10, fontweight='bold')
    ax_d.set_facecolor('#F5F5F5')
    ax_d.spines[['top', 'right']].set_visible(False)

    # ── E: Democratic stress index ────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    stats_yr = cc_by_party_yearly(concept, years)
    dsi = demo_stress_index(stats_yr)

    if len(dsi) > 1:
        ax_e.plot(dsi.index.astype(float), dsi.values, 'o-',
                  color='#8B1A1A', lw=2.2, ms=6)
        ax_e.fill_between(dsi.index.astype(float), dsi.values, alpha=0.15, color='#8B1A1A')
        zmk, pmk = mann_kendall(dsi.values)
        sig_str = '***' if pmk < 0.01 else ('**' if pmk < 0.05 else ('*' if pmk < 0.10 else 'n.s.'))
        ax_e.set_title(f'E. Democratic Stress Index\nMK: Z={zmk:.2f}, p={pmk:.3f} {sig_str}',
                       fontsize=10, fontweight='bold')
        ax_e.set_xlabel('Year', fontsize=9)
        ax_e.set_ylabel('cc_neg / (dr_pos+ε)', fontsize=9)
        for ev_yr, ev_lbl in events:
            ax_e.axvline(ev_yr, color='#555', lw=1.1, ls='--', alpha=0.65)
    else:
        dsi_val = dsi.values[0] if len(dsi) > 0 else np.nan
        ax_e.bar([0], [dsi_val], color='#8B1A1A', width=0.5)
        ax_e.set_xticks([0]); ax_e.set_xticklabels(['2015'])
        ax_e.set_title('E. Democratic Stress Index\n(Single Year)', fontsize=10, fontweight='bold')

    ax_e.set_facecolor('#F5F5F5')
    ax_e.spines[['top', 'right']].set_visible(False)

    # ── F: AKP vs opposition cc_neg ──────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    if len(stats_yr) > 1:
        # Normalize by mention count
        cc_gov_rate = stats_yr['cc_neg_gov'] / (stats_yr['cc_men_gov'] + 0.001)
        cc_opp_rate = stats_yr['cc_neg_opp'] / (stats_yr['cc_men_opp'] + 0.001)
        xs_f = stats_yr.index.astype(float)
        ax_f.plot(xs_f, cc_gov_rate.values, 'o-', color='#E63329', lw=2,
                  ms=5, label='AKP (Gov)')
        ax_f.plot(xs_f, cc_opp_rate.values, 's--', color='#E87722', lw=2,
                  ms=5, label='Opposition')
        ax_f.set_xlabel('Year', fontsize=9)
        ax_f.set_ylabel('cc_neg / cc_mentions', fontsize=9)
        ax_f.legend(fontsize=8)
        for ev_yr, ev_lbl in events:
            ax_f.axvline(ev_yr, color='#555', lw=1.1, ls='--', alpha=0.65)
    else:
        gov_v = stats_yr['cc_neg_gov'].values[0] if len(stats_yr) > 0 else 0
        opp_v = stats_yr['cc_neg_opp'].values[0] if len(stats_yr) > 0 else 0
        ax_f.bar(['AKP', 'Opp.'], [gov_v, opp_v],
                 color=['#E63329', '#E87722'], alpha=0.8)
        ax_f.set_ylabel('cc_neg hits', fontsize=9)

    ax_f.set_title('F. AKP vs Opposition: Constitutional Conflict Negativity',
                   fontsize=10, fontweight='bold')
    ax_f.set_facecolor('#F5F5F5')
    ax_f.spines[['top', 'right']].set_visible(False)

    outpath = f'{outdir}/term_{term}_comprehensive_analysis.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {outpath}')
    return gini


# ── figure 2: institutional conflict ────────────────────────────────────────

def plot_institutional_conflict(term, data, meta, outdir):
    speech  = data['speech']
    concept = data['concept']
    years   = meta['years']
    label   = meta['label']
    events  = meta['events']

    simplified = (len(speech) < 200)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(f'PoliEcoTR — {label}: Institutional Conflict & Democratic Stress',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    stats_yr = cc_by_party_yearly(concept, years)

    # ── A: cc_neg trend all parties stacked ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if len(stats_yr) > 1:
        xs = stats_yr.index.astype(float)
        ax_a.bar(xs - 0.2, stats_yr['cc_neg_gov'], width=0.35, color='#E63329',
                 alpha=0.85, label='AKP (Gov)')
        ax_a.bar(xs + 0.2, stats_yr['cc_neg_opp'], width=0.35, color='#E87722',
                 alpha=0.85, label='Opposition')
        ax_a.set_xlabel('Year', fontsize=9)
        ax_a.set_ylabel('Constitutional Conflict Neg. Hits', fontsize=9)
        ax_a.legend(fontsize=9)
        ax_a.set_xticks(list(xs))
        ax_a.set_xticklabels([str(int(y)) for y in xs], fontsize=8)
        for ev_yr, ev_lbl in events:
            ax_a.axvline(ev_yr, color='#333', lw=1.2, ls='--', alpha=0.65)
            ax_a.text(ev_yr, ax_a.get_ylim()[1]*0.95, ev_lbl,
                      ha='center', va='top', fontsize=7, color='#333')

        # Wilcoxon or t-test AKP vs opp
        if len(stats_yr) >= 4:
            t_ab, p_ab = ttest_ind(stats_yr['cc_neg_gov'], stats_yr['cc_neg_opp'])
            sig_str = '***' if p_ab < 0.01 else ('**' if p_ab < 0.05 else ('*' if p_ab < 0.10 else 'n.s.'))
            ax_a.set_title(f'A. cc_neg: AKP vs Opposition\nt={t_ab:.2f}, p={p_ab:.3f} {sig_str}',
                           fontsize=10, fontweight='bold')
        else:
            ax_a.set_title('A. cc_neg: AKP vs Opposition', fontsize=10, fontweight='bold')
    else:
        gov_v = stats_yr['cc_neg_gov'].values[0] if len(stats_yr) > 0 else 0
        opp_v = stats_yr['cc_neg_opp'].values[0] if len(stats_yr) > 0 else 0
        ax_a.bar(['AKP', 'Opposition'], [gov_v, opp_v], color=['#E63329','#E87722'])
        ax_a.set_title('A. cc_neg: AKP vs Opposition\n(Single Year)', fontsize=10, fontweight='bold')

    ax_a.set_facecolor('#F5F5F5')
    ax_a.spines[['top', 'right']].set_visible(False)

    # ── B: cc_neg / mention_count normalized rate ───────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if len(stats_yr) > 1:
        xs = stats_yr.index.astype(float)
        gov_rate = stats_yr['cc_neg_gov'] / (stats_yr['cc_men_gov'] + 0.001)
        opp_rate = stats_yr['cc_neg_opp'] / (stats_yr['cc_men_opp'] + 0.001)

        # Mann-Kendall on opposition rate (usually stronger trend)
        zmk_opp, pmk_opp = mann_kendall(opp_rate.values)
        zmk_gov, pmk_gov = mann_kendall(gov_rate.values)

        ax_b.plot(xs, gov_rate.values, 'o-', color='#E63329', lw=2.2, ms=6,
                  label=f'AKP (MK p={pmk_gov:.3f})')
        ax_b.plot(xs, opp_rate.values, 's--', color='#E87722', lw=2.2, ms=6,
                  label=f'Opp. (MK p={pmk_opp:.3f})')
        ax_b.fill_between(xs, gov_rate.values, alpha=0.10, color='#E63329')
        ax_b.fill_between(xs, opp_rate.values, alpha=0.10, color='#E87722')
        ax_b.set_xlabel('Year', fontsize=9)
        ax_b.set_ylabel('cc_neg / cc_mentions (normalized)', fontsize=9)
        ax_b.legend(fontsize=8)
        ax_b.set_xticks(list(xs))
        ax_b.set_xticklabels([str(int(y)) for y in xs], fontsize=8)
        for ev_yr, ev_lbl in events:
            ax_b.axvline(ev_yr, color='#555', lw=1.1, ls='--', alpha=0.6)
    else:
        ax_b.text(0.5, 0.5, 'Single-year\nterm', ha='center', va='center',
                  transform=ax_b.transAxes, fontsize=12)

    ax_b.set_title('B. Normalized Constitutional Conflict Rate', fontsize=10, fontweight='bold')
    ax_b.set_facecolor('#F5F5F5')
    ax_b.spines[['top', 'right']].set_visible(False)

    # ── C: Democratic stress index ────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    dsi = demo_stress_index(stats_yr)

    if len(dsi) > 1:
        xs = dsi.index.astype(float)
        zmk, pmk = mann_kendall(dsi.values)
        sig_str = '***' if pmk < 0.01 else ('**' if pmk < 0.05 else ('*' if pmk < 0.10 else 'n.s.'))
        ax_c.plot(xs, dsi.values, 'o-', color='#8B1A1A', lw=2.5, ms=7, zorder=3)
        ax_c.fill_between(xs, dsi.values, alpha=0.18, color='#8B1A1A')

        # Linear trend overlay
        if len(xs) >= 3:
            slope, intercept, *_ = stats.linregress(xs, dsi.values)
            trend_y = slope * xs + intercept
            ax_c.plot(xs, trend_y, 'r--', lw=1.5, alpha=0.7, label=f'Trend (β={slope:.4f})')
            ax_c.legend(fontsize=8)

        ax_c.set_title(f'C. Democratic Stress Index\nMann-Kendall: Z={zmk:.2f}, p={pmk:.3f} {sig_str}',
                       fontsize=10, fontweight='bold')
        ax_c.set_xlabel('Year', fontsize=9)
        ax_c.set_ylabel('(cc_neg) / (dr_pos + ε)', fontsize=9)
        ax_c.set_xticks(list(xs))
        ax_c.set_xticklabels([str(int(y)) for y in xs], fontsize=8)
        for ev_yr, ev_lbl in events:
            ax_c.axvline(ev_yr, color='#555', lw=1.1, ls='--', alpha=0.65)
            ymax = dsi.max()
            ax_c.text(ev_yr, ymax * 1.02, ev_lbl, ha='center', va='bottom',
                      fontsize=7, color='#333', clip_on=True)
        # Annotate peak year
        peak_yr = dsi.idxmax()
        ax_c.annotate(f'Peak: {int(peak_yr)}', xy=(float(peak_yr), dsi.max()),
                      xytext=(5, 8), textcoords='offset points',
                      fontsize=8, color='#8B1A1A',
                      arrowprops=dict(arrowstyle='->', color='#8B1A1A', lw=1))
    else:
        dsi_val = dsi.values[0] if len(dsi) > 0 else 0
        ax_c.bar([0], [dsi_val], color='#8B1A1A', width=0.4)
        ax_c.set_xticks([0]); ax_c.set_xticklabels([str(years[0])])
        ax_c.set_title('C. Democratic Stress Index (Single Year)', fontsize=10, fontweight='bold')

    ax_c.set_facecolor('#F5F5F5')
    ax_c.spines[['top', 'right']].set_visible(False)

    # ── D: Frame competition: cc_neg vs eco_neg ──────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    if len(stats_yr) >= 4:
        cc_total  = (stats_yr['cc_neg_gov'] + stats_yr['cc_neg_opp']).values
        eco_total = (stats_yr['eco_neg_gov'] + stats_yr['eco_neg_opp']).values

        r, p_r = pearsonr(cc_total, eco_total)
        sig_str = '***' if p_r < 0.01 else ('**' if p_r < 0.05 else ('*' if p_r < 0.10 else 'n.s.'))

        ax_d.scatter(eco_total, cc_total, s=80, color='#2196F3', zorder=3, alpha=0.9)
        for i, yr in enumerate(stats_yr.index):
            ax_d.annotate(str(int(yr)), (eco_total[i], cc_total[i]),
                          textcoords='offset points', xytext=(5, 3), fontsize=8)
        # Regression line
        if len(eco_total) >= 3:
            sl, ic, *_ = stats.linregress(eco_total, cc_total)
            xs_r = np.linspace(eco_total.min(), eco_total.max(), 50)
            ax_d.plot(xs_r, sl * xs_r + ic, 'r--', lw=1.8, alpha=0.8)

        ax_d.set_xlabel('Economy Neg. Hits (total)', fontsize=9)
        ax_d.set_ylabel('Const. Conflict Neg. Hits (total)', fontsize=9)
        ax_d.set_title(f'D. Frame Competition: Economy ↔ Constitutional Conflict\nr={r:.3f}, p={p_r:.3f} {sig_str}',
                       fontsize=10, fontweight='bold')
    elif len(stats_yr) > 0:
        # Bar comparison for small samples
        cats_d  = ['cc_neg', 'eco_neg', 'dr_pos']
        vals_d  = [
            (stats_yr['cc_neg_gov'] + stats_yr['cc_neg_opp']).sum(),
            (stats_yr['eco_neg_gov'] + stats_yr['eco_neg_opp']).sum(),
            (stats_yr['dr_pos_gov'] + stats_yr['dr_pos_opp']).sum(),
        ]
        ax_d.bar(cats_d, vals_d, color=['#8B1A1A', '#E63329', '#2196F3'], alpha=0.8)
        ax_d.set_ylabel('Total Hits', fontsize=9)
        ax_d.set_title('D. Topic Hit Comparison (Single Year)', fontsize=10, fontweight='bold')
    else:
        ax_d.text(0.5, 0.5, 'No data', ha='center', va='center',
                  transform=ax_d.transAxes, fontsize=12)
        ax_d.set_title('D. Frame Competition', fontsize=10, fontweight='bold')

    ax_d.set_facecolor('#F5F5F5')
    ax_d.spines[['top', 'right']].set_visible(False)

    outpath = f'{outdir}/term_{term}_institutional_conflict.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  → {outpath}')


# ── archive total_yearly_networks ────────────────────────────────────────────

def archive_total_networks(term, fig_dir):
    archive_dir = f'{fig_dir}/archive'
    os.makedirs(archive_dir, exist_ok=True)
    src = f'{fig_dir}/term_{term}_total_yearly_networks.png'
    dst = f'{archive_dir}/term_{term}_total_yearly_networks.png'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'  → archived: term_{term}_total_yearly_networks.png')
    else:
        print(f'  ! not found (skipping archive): term_{term}_total_yearly_networks.png')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('PoliEcoTR — Multi-Term Analysis Builder')
    print('=' * 60)

    for term in [23, 24, 25, 26, 27, 28]:
        meta    = TERM_META[term]
        fig_dir = f'{BASE}/Term_{term}/Figures'
        print(f'\n── Term {term}: {meta["label"]} ──────────────────────────────')

        # Load data
        data = load_csvs(term)
        n_speech = len(data['speech'])
        print(f'  Loaded: {n_speech} speeches, {len(data["concept"])} concept rows')

        # Comprehensive analysis
        gini = plot_comprehensive(term, data, meta, fig_dir)
        print(f'  Gini={gini:.3f}')

        # Institutional conflict (skip for Term 25 limited data but still produce)
        plot_institutional_conflict(term, data, meta, fig_dir)

        # Archive total yearly networks
        archive_total_networks(term, fig_dir)

        # Key stats summary
        concept = data['concept']
        stats_yr = cc_by_party_yearly(concept, meta['years'])
        if len(stats_yr) > 0:
            dsi = demo_stress_index(stats_yr)
            if len(dsi) > 0:
                peak_yr = dsi.idxmax()
                zmk, pmk = mann_kendall(dsi.values) if len(dsi) >= 4 else (np.nan, np.nan)
                print(f'  Demo Stress peak: {int(peak_yr)} ({dsi.max():.3f})')
                print(f'  Mann-Kendall: Z={zmk:.2f}, p={pmk:.3f}' if not np.isnan(zmk) else '  Mann-Kendall: insufficient data')

    print('\n' + '=' * 60)
    print('All done.')


if __name__ == '__main__':
    main()
