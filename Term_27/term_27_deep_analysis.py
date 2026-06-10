"""
Term 27 Deep Analysis — Causal Identification & Political Dynamics
Events: İmamoğlu İstanbul seçim (2019-06), COVID (2020-03), Albayrak istifası (2020-11),
        TL krizi peak (2022-01), Seçimler (2023-05, outside data)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
BASE    = '/Users/selmanyilmaz/PoliEcoTRNetworSci'
TERM    = 27
CSV     = f'{BASE}/Term_{TERM}/CSVs/term_{TERM}_beme_concept_edges.csv'
FIG_DIR = f'{BASE}/Term_{TERM}/Figures'

BG, PBG = '#FAFAFA', '#F5F5F5'
C_PARTY = {
    'Adalet ve Kalkınma Partisi':   '#E63329',
    'Cumhuriyet Halk Partisi':      '#E87722',
    'Halkların Demokratik Partisi': '#6A0DAD',
    'İYİ Parti':                    '#00A0E3',
    'Milliyetçi Hareket Partisi':   '#8B0000',
}

# ── helpers ───────────────────────────────────────────────────────────────────
def build_monthly_panel(concept_df):
    df = concept_df.copy()
    df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
    total_party = df.groupby(['ym','party'])['mention_count'].sum().reset_index(name='total')
    cats = df.groupby(['ym','party','concept_category']).agg(
        mentions=('mention_count','sum'), pos=('beme_positive_hits','sum'), neg=('beme_negative_hits','sum')
    ).reset_index()
    cc  = cats[cats['concept_category']=='constitutional_conflict'][['ym','party','neg','mentions']].rename(columns={'neg':'cc_neg','mentions':'cc_men'})
    eco = cats[cats['concept_category']=='macro_economy'][['ym','party','neg','mentions']].rename(columns={'neg':'eco_neg','mentions':'eco_men'})
    eco_p = cats[cats['concept_category']=='macro_economy'][['ym','party','pos','mentions']].rename(columns={'pos':'eco_pos','mentions':'eco_pos_men'})
    dr  = cats[cats['concept_category']=='democratic_reform'][['ym','party','pos','mentions']].rename(columns={'pos':'dr_pos','mentions':'dr_men'})
    sec = cats[cats['concept_category']=='security_conflict'][['ym','party','neg','mentions']].rename(columns={'neg':'sec_neg','mentions':'sec_men'})
    m = total_party
    for sub, cols in [(cc,['ym','party','cc_neg','cc_men']), (eco,['ym','party','eco_neg','eco_men']),
                      (eco_p,['ym','party','eco_pos','eco_pos_men']),
                      (dr,['ym','party','dr_pos','dr_men']), (sec,['ym','party','sec_neg','sec_men'])]:
        m = m.merge(sub[cols], on=['ym','party'], how='left')
    m = m.fillna(0)
    m['cc_neg_share']  = m['cc_neg']  / (m['total'] + 0.001)
    m['eco_neg_share'] = m['eco_neg'] / (m['total'] + 0.001)
    m['eco_pos_share'] = m['eco_pos'] / (m['total'] + 0.001)
    m['dr_pos_share']  = m['dr_pos']  / (m['total'] + 0.001)
    m['sec_neg_share'] = m['sec_neg'] / (m['total'] + 0.001)
    m['demo_stress']   = m['cc_neg']  / (m['dr_pos'] + 0.01)
    return m

def agg_all_parties(monthly):
    return monthly.groupby('ym').agg(
        cc_neg=('cc_neg','sum'), eco_neg=('eco_neg','sum'), eco_pos=('eco_pos','sum'),
        dr_pos=('dr_pos','sum'), sec_neg=('sec_neg','sum'), total=('total','sum')
    ).reset_index().assign(
        cc_neg_share=lambda d: d['cc_neg']/(d['total']+0.001),
        eco_neg_share=lambda d: d['eco_neg']/(d['total']+0.001),
        eco_pos_share=lambda d: d['eco_pos']/(d['total']+0.001),
        dr_pos_share=lambda d: d['dr_pos']/(d['total']+0.001),
        sec_neg_share=lambda d: d['sec_neg']/(d['total']+0.001),
    ).sort_values('ym')

def sig_stars(p):
    if np.isnan(p): return 'n.s.'
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return 'n.s.'

def setup_ax(ax):
    ax.set_facecolor(PBG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def add_event_vline(ax, xs_ts, ym_str, label, color='gray', ymax=0.92):
    ev = pd.Period(ym_str, 'M').to_timestamp()
    diffs = [abs((t - ev).days) for t in xs_ts]
    if not diffs: return
    idx = int(np.argmin(diffs))
    if idx >= len(xs_ts): return
    ax.axvline(xs_ts[idx], color=color, lw=1.4, ls='--', alpha=0.8)
    ylim = ax.get_ylim()
    ypos = ylim[0] + (ylim[1]-ylim[0])*ymax
    ax.text(xs_ts[idx], ypos, label, fontsize=7, color=color, ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

def event_study(agg_df, event_ym_str, outcome, window=4):
    event_ym = pd.Period(event_ym_str, 'M')
    agg_sorted = agg_df.sort_values('ym').reset_index(drop=True)
    idx = agg_sorted[agg_sorted['ym']==event_ym].index
    if len(idx)==0: return None, None
    i0 = idx[0]
    lo, hi = max(0, i0-window), min(len(agg_sorted)-1, i0+window)
    sub = agg_sorted.iloc[lo:hi+1].copy()
    sub['rel'] = sub.index - i0
    pre_mean = sub[sub['rel']<0][outcome].mean()
    if pre_mean == 0: pre_mean = 0.001
    sub['norm'] = sub[outcome] / pre_mean
    return sub['rel'].values, sub['norm'].values

def ci95(x):
    if len(x) < 2: return 0
    return 1.96 * x.std() / np.sqrt(len(x))

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
monthly = build_monthly_panel(df)
agg = agg_all_parties(monthly)
xs_ts = [p.to_timestamp() for p in agg['ym'].values]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — causal_identification
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Term 27 — Causal Identification Analysis (2018–2023)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Monthly eco_neg + eco_pos (all parties) — narrative battle
ax = axes[0, 0]
setup_ax(ax)
ax.plot(xs_ts, agg['eco_neg_share'].values, color='#C0392B', lw=1.8, label='eco_neg_share')
ax.plot(xs_ts, agg['eco_pos_share'].values, color='#27AE60', lw=1.8, label='eco_pos_share', ls='--')
for ym_str, lbl, col in [('2020-11', 'Albayrak\nİstifa', '#8B0000'), ('2022-01', 'TL Krizi', '#C0392B')]:
    add_event_vline(ax, xs_ts, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('A: Economic Narrative Battle\neco_neg vs eco_pos share — mark eco_neg peak 2022', fontsize=10, fontweight='bold')
ax.set_ylabel('Share of total mentions')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: AKP eco_pos vs CHP eco_neg — divergence
ax = axes[0, 1]
setup_ax(ax)
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
chp_data = monthly[monthly['party']=='Cumhuriyet Halk Partisi'].sort_values('ym')
xs_akp = [p.to_timestamp() for p in akp_data['ym'].values]
xs_chp = [p.to_timestamp() for p in chp_data['ym'].values]
ax.plot(xs_akp, akp_data['eco_pos_share'].values, color='#E63329', lw=1.8, label='AKP eco_pos_share')
ax.plot(xs_chp, chp_data['eco_neg_share'].values, color='#E87722', lw=1.8, ls='--', label='CHP eco_neg_share')
for ym_str, lbl, col in [('2020-11', 'Albayrak\nİstifa', '#8B0000'), ('2022-01', 'TL\nKrizi', '#C0392B')]:
    add_event_vline(ax, xs_akp, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('B: Narrative Battle — AKP Defends vs CHP Attacks\nAKP eco_pos vs CHP eco_neg monthly', fontsize=10, fontweight='bold')
ax.set_ylabel('Share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel C: Structural break at TL krizi (2022-01)
ax = axes[1, 0]
setup_ax(ax)
try:
    import statsmodels.formula.api as smf
    df_ols = agg.copy()
    event_ym_sb = pd.Period('2022-01', 'M')
    df_ols['post'] = (df_ols['ym'] >= event_ym_sb).astype(int)
    df_ols['time_idx'] = range(len(df_ols))
    df_ols2 = df_ols.rename(columns={'eco_neg_share': 'y'})
    mod = smf.ols('y ~ post + time_idx', data=df_ols2).fit(cov_type='HC3')
    beta_post = mod.params['post']
    pval_post = mod.pvalues['post']
    ci = mod.conf_int().loc['post']
    fitted = mod.fittedvalues.values
    ax.plot(xs_ts, agg['eco_neg_share'].values, color='#C0392B', lw=1.8, label='Actual eco_neg_share')
    ax.plot(xs_ts[:len(fitted)], fitted, color='black', lw=1.5, ls='--', label='OLS fitted')
    ev_ts = event_ym_sb.to_timestamp()
    ax.axvline(ev_ts, color='#C0392B', lw=1.5, ls='--')
    stars = sig_stars(pval_post)
    ax.text(0.05, 0.92, f'β_post = {beta_post:.4f}\np = {pval_post:.3f} {stars}\nHC3 SE',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
    ax.legend(fontsize=8)
except Exception as e:
    ax.text(0.5, 0.5, f'OLS error:\n{e}', transform=ax.transAxes, ha='center')
ax.set_title('C: Structural Break — TL Krizi (Jan 2022)\neco_neg_share OLS post coefficient (HC3)', fontsize=10, fontweight='bold')
ax.set_ylabel('eco_neg_share')
ax.tick_params(axis='x', rotation=30)

# Panel D: Event study around Albayrak istifası (2020-11) ±4M: eco_neg normalized
ax = axes[1, 1]
setup_ax(ax)
rel, norm = event_study(agg, '2020-11', 'eco_neg_share', window=4)
if rel is not None:
    colors_bars = ['#AAB7B8' if r < 0 else '#C0392B' for r in rel]
    ax.bar(rel, norm, color=colors_bars, edgecolor='white', width=0.7)
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.axhline(1, color='gray', lw=1, ls=':')
    for r, n in zip(rel, norm):
        ax.text(r, n + 0.03, f'{n:.2f}', ha='center', fontsize=7.5)
ax.set_title('D: Event Study — Albayrak İstifası (Nov 2020) ±4M\neco_neg normalized to pre-event\n(Does resignation reduce eco_neg?)', fontsize=10, fontweight='bold')
ax.set_xlabel('Months relative to event')
ax.set_ylabel('Normalized eco_neg_share')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = f'{FIG_DIR}/term_{TERM}_causal_identification.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig1)
print(f'Saved: {out1}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — political_dynamics
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 11))
fig2.patch.set_facecolor(BG)
fig2.suptitle('Term 27 — Political Dynamics Analysis (2018–2023)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Parliamentary atrophy — monthly total speech count (all parties)
ax = axes2[0, 0]
setup_ax(ax)
df_vol = df.copy()
df_vol['ym'] = pd.to_datetime(df_vol['date']).dt.to_period('M')
speech_monthly = df_vol.groupby('ym')['mention_count'].sum().reset_index()
speech_monthly = speech_monthly.sort_values('ym')
xs_vol = [p.to_timestamp() for p in speech_monthly['ym'].values]
ax.bar(xs_vol, speech_monthly['mention_count'].values, color='#5D6D7E', alpha=0.7, width=25)
# Trend line
y_vals = speech_monthly['mention_count'].values.astype(float)
x_idx  = np.arange(len(y_vals))
m, b, r, p_tr, se = stats.linregress(x_idx, y_vals)
ax.plot(xs_vol, m*x_idx + b, color='#C0392B', lw=2, ls='--', label=f'Trend (r={r:.2f}, p={p_tr:.3f})')
stars_tr = sig_stars(p_tr)
ax.text(0.05, 0.88, f'Trend slope={m:.1f}/mo\nr={r:.3f}, p={p_tr:.3f} {stars_tr}',
        transform=ax.transAxes, fontsize=9, fontweight='bold',
        bbox=dict(fc='white', ec='gray', alpha=0.8, pad=2))
ax.legend(fontsize=8)
ax.set_title('A: Parliamentary Atrophy Under Presidentialization\nMonthly total mention count — declining?', fontsize=10, fontweight='bold')
ax.set_ylabel('Total mention count')
ax.tick_params(axis='x', rotation=30)

# Panel B: Democratic stress index with key events
ax = axes2[0, 1]
setup_ax(ax)
stress_agg = monthly.groupby('ym').agg(cc_neg=('cc_neg','sum'), dr_pos=('dr_pos','sum')).reset_index()
stress_agg['demo_stress'] = stress_agg['cc_neg'] / (stress_agg['dr_pos'] + 0.01)
stress_agg = stress_agg.sort_values('ym')
xs_s = [p.to_timestamp() for p in stress_agg['ym'].values]
ax.fill_between(xs_s, stress_agg['demo_stress'].values, alpha=0.25, color='#C0392B')
ax.plot(xs_s, stress_agg['demo_stress'].values, color='#C0392B', lw=1.8)
for ym_str, lbl, col in [('2019-06', 'İmamoğlu\nSeçim', '#2980B9'),
                          ('2020-03', 'COVID', '#27AE60'),
                          ('2022-01', 'TL\nKrizi', '#C0392B')]:
    add_event_vline(ax, xs_s, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('B: Democratic Stress Index (cc_neg/dr_pos)\nwith İmamoğlu, COVID, TL Krizi', fontsize=10, fontweight='bold')
ax.set_ylabel('Demo Stress Index')
ax.tick_params(axis='x', rotation=30)

# Panel C: AKP eco_pos by fiscal quarter — minister periods
ax = axes2[1, 0]
setup_ax(ax)
df_akp = df[df['party']=='Adalet ve Kalkınma Partisi'].copy()
df_akp['ym_pd'] = pd.to_datetime(df_akp['date']).dt.to_period('M')
df_akp['quarter'] = pd.to_datetime(df_akp['date']).dt.to_period('Q')
eco_akp = df_akp[df_akp['concept_category']=='macro_economy'].groupby('quarter').agg(
    pos=('beme_positive_hits','sum'), total=('mention_count','sum')
).reset_index()
eco_akp['eco_pos_share'] = eco_akp['pos'] / (eco_akp['total'] + 0.001)
eco_akp = eco_akp.sort_values('quarter')
xs_q = list(range(len(eco_akp)))
q_lbls = [str(q) for q in eco_akp['quarter'].values]
# Color by minister period: Albayrak (before 2020-Q4), Nebati (2020-Q4 to 2023-Q1), Şimşek (2023-Q2+)
albayrak_end = pd.Period('2020Q4', 'Q')
nebati_end   = pd.Period('2023Q1', 'Q')
bar_cols_q = []
for q in eco_akp['quarter'].values:
    if q < albayrak_end:
        bar_cols_q.append('#E63329')
    elif q <= nebati_end:
        bar_cols_q.append('#F5A623')
    else:
        bar_cols_q.append('#27AE60')
ax.bar(xs_q, eco_akp['eco_pos_share'].values, color=bar_cols_q, edgecolor='white', alpha=0.85)
ax.set_xticks(xs_q[::2])
ax.set_xticklabels(q_lbls[::2], rotation=45, fontsize=7)
# Legend
ax.bar([0], [0], color='#E63329', label='Albayrak')
ax.bar([0], [0], color='#F5A623', label='Nebati')
ax.bar([0], [0], color='#27AE60', label='Şimşek')
ax.legend(fontsize=8)
ax.set_title('C: AKP eco_pos_share by Quarter — Minister Periods\nAlbayrak / Nebati / Şimşek', fontsize=10, fontweight='bold')
ax.set_ylabel('AKP eco_pos_share')

# Panel D: Cross-topic competition: cc_neg vs eco_neg Pearson r
ax = axes2[1, 1]
setup_ax(ax)
x = agg['eco_neg_share'].values
y = agg['cc_neg_share'].values
mask = np.isfinite(x) & np.isfinite(y)
x_f, y_f = x[mask], y[mask]
r, pval = stats.pearsonr(x_f, y_f)
ax.scatter(x_f, y_f, color='#5D6D7E', alpha=0.6, s=40, edgecolors='white', lw=0.5)
m_fit, b_fit = np.polyfit(x_f, y_f, 1)
xline = np.linspace(x_f.min(), x_f.max(), 100)
ax.plot(xline, m_fit*xline + b_fit, color='black', lw=1.5, ls='--')
stars = sig_stars(pval)
ax.text(0.05, 0.92, f'Pearson r = {r:.3f}\np = {pval:.3f} {stars}\nn = {mask.sum()}',
        transform=ax.transAxes, fontsize=9, fontweight='bold',
        bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
ax.set_xlabel('eco_neg_share')
ax.set_ylabel('cc_neg_share')
ax.set_title('D: Cross-Topic Competition\ncc_neg vs eco_neg — does economic crisis crowd out\nconstitutional discourse?', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = f'{FIG_DIR}/term_{TERM}_political_dynamics.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('Term 27 done.')
