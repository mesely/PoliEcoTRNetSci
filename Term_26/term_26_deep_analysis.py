"""
Term 26 Deep Analysis — Causal Identification & Political Dynamics
Events: 15 Temmuz (2016-07), OHAL (2016-07), Anayasa Ref (2017-04), HDP tutuklamalar (2016-11)
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
TERM    = 26
CSV     = f'{BASE}/Term_{TERM}/CSVs/term_{TERM}_beme_concept_edges.csv'
FIG_DIR = f'{BASE}/Term_{TERM}/Figures'

# ── colours ──────────────────────────────────────────────────────────────────
C = {
    'Adalet ve Kalkınma Partisi':   '#E63329',
    'Cumhuriyet Halk Partisi':      '#E87722',
    'Halkların Demokratik Partisi': '#6A0DAD',
    'Milliyetçi Hareket Partisi':   '#8B0000',
    'İYİ Parti':                    '#00A0E3',
}
BG, PBG = '#FAFAFA', '#F5F5F5'
COLORS_CAT = {'cc_neg': '#C0392B', 'eco_neg': '#2980B9', 'dr_pos': '#27AE60', 'sec_neg': '#8E44AD'}

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
    dr  = cats[cats['concept_category']=='democratic_reform'][['ym','party','pos','mentions']].rename(columns={'pos':'dr_pos','mentions':'dr_men'})
    sec = cats[cats['concept_category']=='security_conflict'][['ym','party','neg','mentions']].rename(columns={'neg':'sec_neg','mentions':'sec_men'})
    m = total_party
    for sub, cols in [(cc,['ym','party','cc_neg','cc_men']), (eco,['ym','party','eco_neg','eco_men']),
                      (dr,['ym','party','dr_pos','dr_men']), (sec,['ym','party','sec_neg','sec_men'])]:
        m = m.merge(sub[cols], on=['ym','party'], how='left')
    m = m.fillna(0)
    m['cc_neg_share']  = m['cc_neg']  / (m['total'] + 0.001)
    m['eco_neg_share'] = m['eco_neg'] / (m['total'] + 0.001)
    m['dr_pos_share']  = m['dr_pos']  / (m['total'] + 0.001)
    m['sec_neg_share'] = m['sec_neg'] / (m['total'] + 0.001)
    m['demo_stress']   = m['cc_neg']  / (m['dr_pos'] + 0.01)
    return m

def agg_all_parties(monthly):
    return monthly.groupby('ym').agg(
        cc_neg=('cc_neg','sum'), eco_neg=('eco_neg','sum'), dr_pos=('dr_pos','sum'),
        sec_neg=('sec_neg','sum'), total=('total','sum')
    ).reset_index().assign(
        cc_neg_share=lambda d: d['cc_neg']/(d['total']+0.001),
        eco_neg_share=lambda d: d['eco_neg']/(d['total']+0.001),
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
    idx = int(np.argmin(diffs))
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
fig1.suptitle('Term 26 — Causal Identification Analysis (2015–2018)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Monthly cc_neg + dr_pos + sec_neg (all parties)
ax = axes[0, 0]
setup_ax(ax)
ax.plot(xs_ts, agg['cc_neg_share'].values, color=COLORS_CAT['cc_neg'], lw=1.8, label='cc_neg_share')
ax.plot(xs_ts, agg['dr_pos_share'].values, color=COLORS_CAT['dr_pos'], lw=1.8, label='dr_pos_share', ls='--')
ax.plot(xs_ts, agg['sec_neg_share'].values, color=COLORS_CAT['sec_neg'], lw=1.8, label='sec_neg_share', ls=':')
for ym_str, lbl, col in [('2016-07', '15 Tem.', '#C0392B'), ('2017-04', 'Anayasa\nRef.', '#27AE60')]:
    add_event_vline(ax, xs_ts, ym_str, lbl, color=col, ymax=0.88)
ax.set_title('A: Constitutional Conflict, Democratic Reform, Security\n(OHAL suppression → cc_neg explosion before referendum)', fontsize=10, fontweight='bold')
ax.set_ylabel('Share of total mentions')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: HDP monthly total mention count — silencing visible
ax = axes[0, 1]
setup_ax(ax)
hdp_monthly = df.copy()
hdp_monthly['ym'] = pd.to_datetime(hdp_monthly['date']).dt.to_period('M')
hdp_counts = hdp_monthly[hdp_monthly['party']=='Halkların Demokratik Partisi'].groupby('ym')['mention_count'].sum().reset_index()
hdp_counts = hdp_counts.sort_values('ym')
xs_hdp = [p.to_timestamp() for p in hdp_counts['ym'].values]
ax.bar(xs_hdp, hdp_counts['mention_count'].values, color='#6A0DAD', alpha=0.7, width=25)
for ym_str, lbl, col in [('2016-07', '15 Tem.', '#C0392B'), ('2016-11', 'HDP\nTutuklama', '#E63329')]:
    ev_ts = pd.Period(ym_str, 'M').to_timestamp()
    ax.axvline(ev_ts, color=col, lw=1.5, ls='--')
    ylim = ax.get_ylim()
    ax.text(ev_ts, ylim[0] + (ylim[1]-ylim[0])*0.85, lbl, fontsize=8, color=col, ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('B: HDP Voice Collapse — Monthly Total Mentions\n(Silencing after 15 Temmuz & Nov 2016 arrests)', fontsize=10, fontweight='bold')
ax.set_ylabel('Total mention count')
ax.tick_params(axis='x', rotation=30)

# Panel C: Structural break OLS at Anayasa Ref campaign start (2017-01)
ax = axes[1, 0]
setup_ax(ax)
try:
    import statsmodels.formula.api as smf
    df_ols = agg.copy()
    event_ym_sb = pd.Period('2017-01', 'M')
    df_ols['post'] = (df_ols['ym'] >= event_ym_sb).astype(int)
    df_ols['time_idx'] = range(len(df_ols))
    df_ols2 = df_ols.rename(columns={'cc_neg_share': 'y'})
    mod = smf.ols('y ~ post + time_idx', data=df_ols2).fit(cov_type='HC3')
    beta_post = mod.params['post']
    pval_post = mod.pvalues['post']
    ci = mod.conf_int().loc['post']
    ax.bar(['post coef'], [beta_post], color='#C0392B' if beta_post > 0 else '#2980B9',
           edgecolor='black', width=0.4)
    ax.errorbar(['post coef'], [beta_post],
                yerr=[[beta_post - ci[0]], [ci[1] - beta_post]],
                fmt='none', color='black', capsize=8, lw=2)
    ax.axhline(0, color='black', lw=1)
    stars = sig_stars(pval_post)
    ypos_ann = beta_post + (ci[1]-beta_post)*1.15 if beta_post > 0 else beta_post - abs(beta_post)*0.5
    ax.annotate(f'β={beta_post:.4f}\np={pval_post:.3f} {stars}\nHC3 SE',
                xy=(0, beta_post), xytext=(0, ypos_ann),
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
except Exception as e:
    ax.text(0.5, 0.5, f'OLS error:\n{e}', transform=ax.transAxes, ha='center')
ax.set_title('C: Structural Break — Anayasa Ref Campaign (Jan 2017)\ncc_neg_share post coefficient (HC3)', fontsize=10, fontweight='bold')
ax.set_ylabel('Coefficient')

# Panel D: Three-period cc_neg comparison: pre-coup / OHAL / post-OHAL
ax = axes[1, 1]
setup_ax(ax)
pre_coup_ym  = pd.Period('2016-07', 'M')   # < 2016-07
ohal_end_ym  = pd.Period('2017-07', 'M')   # 2016-07 to 2017-07
# post: >= 2017-07

period_labels = ['Pre-Coup\n(<2016-07)', 'OHAL\n(2016-07–2017-07)', 'Post-OHAL\n(>2017-07)']
party_groups = [
    ('Adalet ve Kalkınma Partisi', 'AKP', '#E63329'),
    ('Cumhuriyet Halk Partisi',    'CHP+Opp', '#E87722'),
]

n_periods = 3
n_parties = len(party_groups)
width = 0.3
x = np.arange(n_periods)

for pi, (party, lbl, col) in enumerate(party_groups):
    pdata = monthly[monthly['party']==party].sort_values('ym')
    vals, errs = [], []
    for mask_fn in [
        lambda d: d[d['ym'] < pre_coup_ym],
        lambda d: d[(d['ym'] >= pre_coup_ym) & (d['ym'] < ohal_end_ym)],
        lambda d: d[d['ym'] >= ohal_end_ym],
    ]:
        sub = mask_fn(pdata)
        v = sub['cc_neg_share'].values
        vals.append(v.mean() if len(v) > 0 else 0)
        errs.append(ci95(v))
    offset = (pi - (n_parties-1)/2) * (width + 0.05)
    ax.bar(x + offset, vals, width=width, color=col, edgecolor='black', label=lbl, alpha=0.8)
    ax.errorbar(x + offset, vals, yerr=errs, fmt='none', color='black', capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(period_labels, fontsize=9)
ax.set_title('D: cc_neg_share — Three Periods\nPre-Coup / OHAL / Post-OHAL with 95% CI', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean cc_neg_share')
ax.legend(fontsize=8)

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
fig2.suptitle('Term 26 — Political Dynamics Analysis (2015–2018)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: "Suppression then explosion" — cc_neg with period shading
ax = axes2[0, 0]
setup_ax(ax)
pre_coup_ts  = pd.Period('2016-07', 'M').to_timestamp()
ohal_end_ts  = pd.Period('2017-07', 'M').to_timestamp()
x_min = xs_ts[0]
x_max = xs_ts[-1]
# Shade periods
ax.axvspan(x_min, pre_coup_ts, alpha=0.08, color='#27AE60', label='Pre-coup')
ax.axvspan(pre_coup_ts, ohal_end_ts, alpha=0.12, color='#C0392B', label='OHAL')
ax.axvspan(ohal_end_ts, x_max, alpha=0.08, color='#2980B9', label='Post-OHAL/Ref')
ax.plot(xs_ts, agg['cc_neg_share'].values, color=COLORS_CAT['cc_neg'], lw=1.8, zorder=3)
ax.axvline(pre_coup_ts, color='#C0392B', lw=1.5, ls='--')
ax.axvline(ohal_end_ts, color='#2980B9', lw=1.5, ls='--')
ax.set_title('A: Suppression Then Explosion\ncc_neg with coup/OHAL/post-OHAL shading', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share (all parties)')
ax.legend(fontsize=8, loc='upper left')
ax.tick_params(axis='x', rotation=30)

# Panel B: Party speech volume — monthly total mentions stacked (HDP disappears)
ax = axes2[0, 1]
setup_ax(ax)
df_vol = df.copy()
df_vol['ym'] = pd.to_datetime(df_vol['date']).dt.to_period('M')
party_vol = df_vol.groupby(['ym','party'])['mention_count'].sum().reset_index()
major_parties = ['Adalet ve Kalkınma Partisi', 'Cumhuriyet Halk Partisi',
                 'Halkların Demokratik Partisi', 'Milliyetçi Hareket Partisi']
all_ym = sorted(party_vol['ym'].unique())
xs_all = [p.to_timestamp() for p in all_ym]
bottoms = np.zeros(len(all_ym))
cols_vol = ['#E63329', '#E87722', '#6A0DAD', '#8B0000']
lbls_vol = ['AKP', 'CHP', 'HDP', 'MHP']
for party, col, lbl in zip(major_parties, cols_vol, lbls_vol):
    vals = []
    for ym in all_ym:
        sub = party_vol[(party_vol['ym']==ym) & (party_vol['party']==party)]
        vals.append(sub['mention_count'].sum() if len(sub) > 0 else 0)
    vals = np.array(vals, dtype=float)
    ax.bar(xs_all, vals, bottom=bottoms, color=col, label=lbl, width=25, alpha=0.85)
    bottoms += vals
for ym_str, lbl, col in [('2016-07', '15 Tem.', 'black'), ('2016-11', 'HDP\nArr.', 'white')]:
    ev_ts = pd.Period(ym_str, 'M').to_timestamp()
    ax.axvline(ev_ts, color=col, lw=1.5, ls='--')
ax.set_title('B: Parliamentary Voice — Monthly Speech Volume by Party\n(HDP disappears after Nov 2016)', fontsize=10, fontweight='bold')
ax.set_ylabel('Total mention count')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel C: AKP sec_neg vs CHP dr_pos after coup
ax = axes2[1, 0]
setup_ax(ax)
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
chp_data = monthly[monthly['party']=='Cumhuriyet Halk Partisi'].sort_values('ym')
xs_akp = [p.to_timestamp() for p in akp_data['ym'].values]
xs_chp = [p.to_timestamp() for p in chp_data['ym'].values]
ax.plot(xs_akp, akp_data['sec_neg_share'].values, color='#E63329', lw=1.8, label='AKP sec_neg_share')
ax.plot(xs_chp, chp_data['dr_pos_share'].values,  color='#E87722', lw=1.8, ls='--', label='CHP dr_pos_share')
ax.axvline(pre_coup_ts, color='gray', lw=1.5, ls='--')
ylim = ax.get_ylim()
ax.text(pre_coup_ts, ylim[0] + (ylim[1]-ylim[0])*0.85, '15 Tem.', fontsize=8, color='gray', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('C: AKP Securitizes vs CHP Defends Democracy\nAKP sec_neg vs CHP dr_pos post-coup', fontsize=10, fontweight='bold')
ax.set_ylabel('Share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel D: Anayasa Ref campaign: cc_neg breakdown by party (2016-10 to 2017-04)
ax = axes2[1, 1]
setup_ax(ax)
camp_start = pd.Period('2016-10', 'M')
camp_end   = pd.Period('2017-04', 'M')
parties_camp = [
    ('Adalet ve Kalkınma Partisi',   '#E63329', 'AKP'),
    ('Cumhuriyet Halk Partisi',      '#E87722', 'CHP'),
    ('Halkların Demokratik Partisi', '#6A0DAD', 'HDP'),
    ('Milliyetçi Hareket Partisi',   '#8B0000', 'MHP'),
]
for party, col, lbl in parties_camp:
    pdata = monthly[monthly['party']==party].sort_values('ym')
    pdata = pdata[(pdata['ym'] >= camp_start) & (pdata['ym'] <= camp_end)]
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['cc_neg_share'].values, color=col, lw=2, marker='o', ms=5, label=lbl)
ax.set_title('D: Anayasa Ref Campaign — cc_neg by Party\n(Oct 2016 – Apr 2017): Who drives the spike?', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = f'{FIG_DIR}/term_{TERM}_political_dynamics.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('Term 26 done.')
