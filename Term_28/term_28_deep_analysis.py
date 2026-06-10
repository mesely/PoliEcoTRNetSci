"""
Term 28 Deep Analysis — Causal Identification & Political Dynamics
Events: Gaza/Filistin (2023-10), Yerel Seçim AKP yenilgisi (2024-03),
        İmamoğlu tutuklanması (2025-03), Post-tutuklama protestolar (2025-04)
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
TERM    = 28
CSV     = f'{BASE}/Term_{TERM}/CSVs/term_{TERM}_beme_concept_edges.csv'
FIG_DIR = f'{BASE}/Term_{TERM}/Figures'

BG, PBG = '#FAFAFA', '#F5F5F5'
C_PARTY = {
    'Adalet ve Kalkınma Partisi':                '#E63329',
    'Cumhuriyet Halk Partisi':                   '#E87722',
    'Halkların Eşitlik ve Demokrasi Partisi':    '#6A0DAD',
    'İYİ Parti':                                 '#00A0E3',
    'Milliyetçi Hareket Partisi':                '#8B0000',
    'YENİ YOL Partisi':                          '#5D6D7E',
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
fig1.suptitle('Term 28 — Causal Identification Analysis (2023–2026)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Monthly cc_neg_share AKP vs CHP with event markers
ax = axes[0, 0]
setup_ax(ax)
for party, col, lbl in [
    ('Adalet ve Kalkınma Partisi', '#E63329', 'AKP'),
    ('Cumhuriyet Halk Partisi',    '#E87722', 'CHP'),
]:
    pdata = monthly[monthly['party']==party].sort_values('ym')
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['cc_neg_share'].values, color=col, lw=1.8, label=lbl)
for ym_str, lbl, col in [('2024-03', 'Yerel\nSeçim', '#27AE60'), ('2025-03', 'İmamoğlu\nTutuklama', '#C0392B')]:
    add_event_vline(ax, xs_ts, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('A: cc_neg_share — AKP vs CHP\nwith Yerel Seçim + İmamoğlu tutuklanması', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: Structural break at İmamoğlu tutuklanması (2025-03): cc_neg fitted vs actual
ax = axes[0, 1]
setup_ax(ax)
try:
    import statsmodels.formula.api as smf
    df_ols = agg.copy()
    event_ym_sb = pd.Period('2025-03', 'M')
    df_ols['post'] = (df_ols['ym'] >= event_ym_sb).astype(int)
    df_ols['time_idx'] = range(len(df_ols))
    df_ols2 = df_ols.rename(columns={'cc_neg_share': 'y'})
    mod = smf.ols('y ~ post + time_idx', data=df_ols2).fit(cov_type='HC3')
    beta_post = mod.params['post']
    pval_post = mod.pvalues['post']
    fitted = mod.fittedvalues.values
    ax.plot(xs_ts, agg['cc_neg_share'].values, color='#C0392B', lw=1.8, label='Actual cc_neg_share')
    ax.plot(xs_ts[:len(fitted)], fitted, color='black', lw=1.5, ls='--', label='OLS fitted')
    ev_ts = event_ym_sb.to_timestamp()
    ax.axvline(ev_ts, color='#C0392B', lw=2, ls='--')
    ylim = ax.get_ylim()
    ax.text(ev_ts, ylim[0] + (ylim[1]-ylim[0])*0.85, 'İmamoğlu\nTutuklama', fontsize=8, color='#C0392B', ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    stars = sig_stars(pval_post)
    ax.text(0.05, 0.92, f'β_post = {beta_post:.4f}\np = {pval_post:.3f} {stars}\nHC3 SE',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
    ax.legend(fontsize=8)
except Exception as e:
    ax.text(0.5, 0.5, f'OLS error:\n{e}', transform=ax.transAxes, ha='center')
ax.set_title('B: Structural Break — İmamoğlu Tutuklanması (Mar 2025)\ncc_neg_share OLS fitted vs actual', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')
ax.tick_params(axis='x', rotation=30)

# Panel C: AKP cc_neg_share month-by-month 2024-12 to 2025-06 — zoom spike
ax = axes[1, 0]
setup_ax(ax)
zoom_start = pd.Period('2024-12', 'M')
zoom_end_p = pd.Period('2025-06', 'M')
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
akp_zoom = akp_data[(akp_data['ym'] >= zoom_start) & (akp_data['ym'] <= zoom_end_p)]
if len(akp_zoom) > 0:
    xs_z = [p.to_timestamp() for p in akp_zoom['ym'].values]
    vals_z = akp_zoom['cc_neg_share'].values
    bar_colors_z = ['#E63329' if p >= pd.Period('2025-03','M') else '#AAB7B8' for p in akp_zoom['ym'].values]
    ax.bar(xs_z, vals_z, color=bar_colors_z, edgecolor='black', width=22)
    for x_b, v, ym_b in zip(xs_z, vals_z, akp_zoom['ym'].values):
        ax.text(x_b, v + 0.003, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold' if ym_b >= pd.Period('2025-03','M') else 'normal')
    # Annotate 2025-04 spike
    apr_mask = akp_zoom['ym'] == pd.Period('2025-04','M')
    if apr_mask.any():
        apr_val = akp_zoom[apr_mask]['cc_neg_share'].values[0]
        apr_ts  = pd.Period('2025-04','M').to_timestamp()
        ax.annotate(f'AKP 2025-04\n= {apr_val:.3f}\n(conviction\njustification)',
                    xy=(apr_ts, apr_val),
                    xytext=(apr_ts, apr_val * 0.7),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=8, ha='center',
                    bbox=dict(fc='white', ec='gray', alpha=0.8, pad=2))
ax.set_title('C: AKP cc_neg_share Zoom — Dec 2024 to Jun 2025\n(Red = post-tutuklama)', fontsize=10, fontweight='bold')
ax.set_ylabel('AKP cc_neg_share')
ax.tick_params(axis='x', rotation=30)

# Panel D: CHP dr_pos_share 2024-06 to 2025-06
ax = axes[1, 1]
setup_ax(ax)
chp_data = monthly[monthly['party']=='Cumhuriyet Halk Partisi'].sort_values('ym')
dr_start = pd.Period('2024-06', 'M')
dr_end   = pd.Period('2025-06', 'M')
# check what end of data is
max_ym = chp_data['ym'].max()
dr_end = min(dr_end, max_ym)
chp_dr = chp_data[(chp_data['ym'] >= dr_start) & (chp_data['ym'] <= dr_end)]
if len(chp_dr) > 0:
    xs_dr = [p.to_timestamp() for p in chp_dr['ym'].values]
    ax.plot(xs_dr, chp_dr['dr_pos_share'].values, color='#E87722', lw=2, marker='o', ms=5)
    ax.fill_between(xs_dr, chp_dr['dr_pos_share'].values, alpha=0.2, color='#E87722')
    # mark tutuklama
    tut_ts = pd.Period('2025-03','M').to_timestamp()
    if xs_dr[0] <= tut_ts <= xs_dr[-1]:
        ax.axvline(tut_ts, color='#C0392B', lw=2, ls='--')
        ylim = ax.get_ylim()
        ax.text(tut_ts, ylim[0] + (ylim[1]-ylim[0])*0.85, 'İmamoğlu\nTutuklama', fontsize=8, color='#C0392B', ha='center',
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('D: CHP dr_pos_share — Jun 2024 to Jun 2025\nDemocratic reform surge after tutuklama?', fontsize=10, fontweight='bold')
ax.set_ylabel('CHP dr_pos_share')
ax.tick_params(axis='x', rotation=30)

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
fig2.suptitle('Term 28 — Political Dynamics Analysis (2023–2026)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Yerel Seçim effect: AKP eco_pos vs eco_neg before/after 2024-03
ax = axes2[0, 0]
setup_ax(ax)
seçim_ym = pd.Period('2024-03', 'M')
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
pre_pos  = akp_data[akp_data['ym'] < seçim_ym]['eco_pos_share'].values
post_pos = akp_data[akp_data['ym'] >= seçim_ym]['eco_pos_share'].values
pre_neg  = akp_data[akp_data['ym'] < seçim_ym]['eco_neg_share'].values
post_neg = akp_data[akp_data['ym'] >= seçim_ym]['eco_neg_share'].values

labels = ['eco_pos\npre', 'eco_pos\npost', 'eco_neg\npre', 'eco_neg\npost']
vals_bar = [pre_pos.mean(), post_pos.mean(), pre_neg.mean(), post_neg.mean()]
errs_bar = [ci95(pre_pos), ci95(post_pos), ci95(pre_neg), ci95(post_neg)]
bar_c = ['#27AE60', '#1A8A40', '#C0392B', '#922B21']
bars = ax.bar(labels, vals_bar, color=bar_c, edgecolor='black', width=0.5)
ax.errorbar(labels, vals_bar, yerr=errs_bar, fmt='none', color='black', capsize=6, lw=2)

# t-tests
t_pos, p_pos = stats.ttest_ind(post_pos, pre_pos) if (len(pre_pos)>=2 and len(post_pos)>=2) else (np.nan, np.nan)
t_neg, p_neg = stats.ttest_ind(post_neg, pre_neg) if (len(pre_neg)>=2 and len(post_neg)>=2) else (np.nan, np.nan)
ymax_v = max(vals_bar) * 1.25
ax.text(0.5, ymax_v, f'eco_pos: t={t_pos:.2f}, p={p_pos:.3f} {sig_stars(p_pos)}', ha='center', fontsize=8,
        bbox=dict(fc='lightyellow', ec='gray', alpha=0.8, pad=2))
ax.text(0.5, ymax_v * 0.85, f'eco_neg: t={t_neg:.2f}, p={p_neg:.3f} {sig_stars(p_neg)}', ha='center', fontsize=8,
        bbox=dict(fc='#FADBD8', ec='gray', alpha=0.8, pad=2))
ax.set_title('A: AKP Economic Framing — Before vs After Yerel Seçim\nDoes AKP accept economic failure after losing?', fontsize=10, fontweight='bold')
ax.set_ylabel('AKP mean share')

# Panel B: Gaza solidarity — sec_neg_share monthly with 2023-10 marker
ax = axes2[0, 1]
setup_ax(ax)
for party, col, lbl in [
    ('Adalet ve Kalkınma Partisi',             '#E63329', 'AKP'),
    ('Cumhuriyet Halk Partisi',                '#E87722', 'CHP'),
    ('Halkların Eşitlik ve Demokrasi Partisi', '#6A0DAD', 'HEDEP'),
]:
    pdata = monthly[monthly['party']==party].sort_values('ym')
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['sec_neg_share'].values, color=col, lw=1.8, label=lbl)
gaza_ts = pd.Period('2023-10', 'M').to_timestamp()
ax.axvline(gaza_ts, color='#8E44AD', lw=1.5, ls='--')
ylim = ax.get_ylim()
ax.text(gaza_ts, ylim[0] + (ylim[1]-ylim[0])*0.88, 'Gaza\n2023-10', fontsize=8, color='#8E44AD', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('B: Gaza Solidarity — sec_neg_share by Party\nInternational conflict securitization', fontsize=10, fontweight='bold')
ax.set_ylabel('sec_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel C: Democratic stress index with both main events
ax = axes2[1, 0]
setup_ax(ax)
stress_agg = monthly.groupby('ym').agg(cc_neg=('cc_neg','sum'), dr_pos=('dr_pos','sum')).reset_index()
stress_agg['demo_stress'] = stress_agg['cc_neg'] / (stress_agg['dr_pos'] + 0.01)
stress_agg = stress_agg.sort_values('ym')
xs_s = [p.to_timestamp() for p in stress_agg['ym'].values]
ax.fill_between(xs_s, stress_agg['demo_stress'].values, alpha=0.25, color='#C0392B')
ax.plot(xs_s, stress_agg['demo_stress'].values, color='#C0392B', lw=1.8)
for ym_str, lbl, col in [('2024-03', 'Yerel\nSeçim', '#27AE60'), ('2025-03', 'İmamoğlu\nTutuklama', '#C0392B')]:
    add_event_vline(ax, xs_s, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('C: Democratic Stress Index (cc_neg/dr_pos)\nwith Yerel Seçim and İmamoğlu Tutuklanması', fontsize=10, fontweight='bold')
ax.set_ylabel('Demo Stress Index')
ax.tick_params(axis='x', rotation=30)

# Panel D: Effective number of parties in speech per month — plural or concentrated?
ax = axes2[1, 1]
setup_ax(ax)
df_vol = df.copy()
df_vol['ym'] = pd.to_datetime(df_vol['date']).dt.to_period('M')
party_vol = df_vol.groupby(['ym','party'])['mention_count'].sum().reset_index()

def enp(group):
    total = group['mention_count'].sum()
    if total == 0: return np.nan
    shares = group['mention_count'] / total
    shares = shares[shares > 0]
    entropy = -np.sum(shares * np.log(shares))
    return np.exp(entropy)

enp_monthly = party_vol.groupby('ym').apply(enp).reset_index()
enp_monthly.columns = ['ym', 'enp']
enp_monthly = enp_monthly.sort_values('ym')
xs_enp = [p.to_timestamp() for p in enp_monthly['ym'].values]
enp_vals = enp_monthly['enp'].values

ax.plot(xs_enp, enp_vals, color='#2980B9', lw=1.8)
ax.fill_between(xs_enp, enp_vals, alpha=0.2, color='#2980B9')
# Trend
x_idx = np.arange(len(enp_vals))
mask_enp = np.isfinite(enp_vals)
if mask_enp.sum() >= 4:
    m_e, b_e, r_e, p_e, _ = stats.linregress(x_idx[mask_enp], enp_vals[mask_enp])
    ax.plot(xs_enp, m_e*x_idx + b_e, color='#C0392B', lw=1.5, ls='--',
            label=f'Trend r={r_e:.2f} {sig_stars(p_e)}')
    ax.legend(fontsize=8)

for ym_str, lbl, col in [('2024-03', 'Yerel\nSeçim', '#27AE60'), ('2025-03', 'İmamoğlu\nTutuklama', '#C0392B')]:
    add_event_vline(ax, xs_enp, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('D: Effective Number of Parties in Parliamentary Speech\n(exp(entropy)) — plural or concentrated?', fontsize=10, fontweight='bold')
ax.set_ylabel('ENP (speech diversity)')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = f'{FIG_DIR}/term_{TERM}_political_dynamics.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('Term 28 done.')
