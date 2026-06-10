"""
Term 24 Deep Analysis — Causal Identification & Political Dynamics
Events: Çözüm Süreci (2013-03), Gezi (2013-05), 17-25 Aralık (2013-12), Dershaneler (2014-03)
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
TERM    = 24
CSV     = f'{BASE}/Term_{TERM}/CSVs/term_{TERM}_beme_concept_edges.csv'
FIG_DIR = f'{BASE}/Term_{TERM}/Figures'

# ── colours ──────────────────────────────────────────────────────────────────
C = {
    'Adalet ve Kalkınma Partisi':      '#E63329',
    'Cumhuriyet Halk Partisi':         '#E87722',
    'Milliyetçi Hareket Partisi':      '#8B0000',
    'Halkların Demokratik Partisi':    '#6A0DAD',
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

def structural_break_ols(agg_df, event_ym_str, outcome='cc_neg_share'):
    import statsmodels.formula.api as smf
    df = agg_df.copy()
    event_ym = pd.Period(event_ym_str, 'M')
    df['post'] = (df['ym'] >= event_ym).astype(int)
    df['time_idx'] = range(len(df))
    df_f = df.rename(columns={outcome: 'y'})
    mod = smf.ols('y ~ post + time_idx', data=df_f).fit(cov_type='HC3')
    return mod

def party_prepost(monthly, event_ym_str, party, outcome='cc_neg_share'):
    event_ym = pd.Period(event_ym_str, 'M')
    pdata = monthly[monthly['party']==party].sort_values('ym')
    pre   = pdata[pdata['ym'] < event_ym][outcome].values
    post  = pdata[pdata['ym'] >= event_ym][outcome].values
    if len(pre) < 2 or len(post) < 2:
        return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(post, pre)
    return t, p, pre.mean(), post.mean()

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

events = {
    'cozum':     ('2013-03', 'Çözüm\nSüreci', '#27AE60'),
    'gezi':      ('2013-05', 'Gezi',           '#2980B9'),
    'aralik':    ('2013-12', '17-25\nAralık',  '#C0392B'),
    'dershane':  ('2014-03', 'Dershane\nKarar','#8E44AD'),
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — causal_identification
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Term 24 — Causal Identification Analysis (2011–2015)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Monthly cc_neg + dr_pos + sec_neg all parties, 4 event markers
ax = axes[0, 0]
setup_ax(ax)
ax.plot(xs_ts, agg['cc_neg_share'].values, color=COLORS_CAT['cc_neg'], lw=1.8, label='cc_neg_share')
ax.plot(xs_ts, agg['dr_pos_share'].values, color=COLORS_CAT['dr_pos'], lw=1.8, label='dr_pos_share', ls='--')
ax.plot(xs_ts, agg['sec_neg_share'].values, color=COLORS_CAT['sec_neg'], lw=1.8, label='sec_neg_share', ls=':')
for key, (ym, lbl, col) in events.items():
    add_event_vline(ax, xs_ts, ym, lbl, color=col, ymax=0.88)
ax.set_title('A: Constitutional Conflict vs Democratic Reform vs Security\n(all parties, monthly)', fontsize=10, fontweight='bold')
ax.set_ylabel('Share of total mentions')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: Dual event study — Gezi effect on dr_pos ±4M, 17-25 Aralık on cc_neg ±4M
ax = axes[0, 1]
setup_ax(ax)
rel_gezi, norm_gezi = event_study(agg, '2013-05', 'dr_pos_share', window=4)
rel_aralik, norm_aralik = event_study(agg, '2013-12', 'cc_neg_share', window=4)
if rel_gezi is not None:
    ax.plot(rel_gezi, norm_gezi, color='#2980B9', lw=2, marker='o', ms=5, label='Gezi → dr_pos')
if rel_aralik is not None:
    ax.plot(rel_aralik, norm_aralik, color='#C0392B', lw=2, marker='s', ms=5, label='17-25 Aralık → cc_neg')
ax.axvline(0, color='black', lw=1.5, ls='--')
ax.axhline(1, color='gray', lw=1, ls=':')
ax.set_title('B: Dual Event Study\nGezi→dr_pos vs 17-25 Aralık→cc_neg (±4M)', fontsize=10, fontweight='bold')
ax.set_xlabel('Months relative to event')
ax.set_ylabel('Normalized value (pre-event mean = 1)')
ax.legend(fontsize=8)

# Panel C: AKP & CHP pre/post t-test at BOTH events (4 bars)
ax = axes[1, 0]
setup_ax(ax)
parties_test = [('Adalet ve Kalkınma Partisi', 'AKP'), ('Cumhuriyet Halk Partisi', 'CHP')]
event_tests  = [('2013-05', 'Gezi'), ('2013-12', '17-25 Aralık')]

bar_labels, bar_vals, bar_errs, bar_cols = [], [], [], []
stat_texts = []
x_pos = 0
tick_pos, tick_lbls = [], []

for ym, ev_lbl in event_tests:
    for party, plbl in parties_test:
        t, p, pre_m, post_m = party_prepost(monthly, ym, party, 'cc_neg_share')
        ev_ym = pd.Period(ym, 'M')
        pdata = monthly[monthly['party']==party].sort_values('ym')
        pre_v  = pdata[pdata['ym'] < ev_ym]['cc_neg_share'].values
        post_v = pdata[pdata['ym'] >= ev_ym]['cc_neg_share'].values
        stars = sig_stars(p) if not np.isnan(p) else 'n.s.'
        col_pre  = C.get(party, '#888888')
        col_post = C.get(party, '#888888')
        ax.bar(x_pos,     pre_m,  color=col_pre,  alpha=0.5, edgecolor='black', width=0.4, label=f'{plbl} pre' if x_pos==0 else '')
        ax.bar(x_pos+0.5, post_m, color=col_post, alpha=1.0, edgecolor='black', width=0.4)
        ax.errorbar(x_pos,     pre_m,  yerr=ci95(pre_v),  fmt='none', color='black', capsize=4)
        ax.errorbar(x_pos+0.5, post_m, yerr=ci95(post_v), fmt='none', color='black', capsize=4)
        ymax_here = max(pre_m + ci95(pre_v), post_m + ci95(post_v)) * 1.15
        ax.text(x_pos + 0.25, ymax_here, stars, ha='center', fontsize=9, fontweight='bold')
        tick_pos.append(x_pos + 0.25)
        tick_lbls.append(f'{plbl}\n{ev_lbl}')
        x_pos += 1.5

ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, fontsize=8)
ax.set_title('C: Pre/Post t-test at Gezi & 17-25 Aralık\nAKP & CHP cc_neg_share (light=pre, dark=post)', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean cc_neg_share')

# Panel D: Structural break at 17-25 Aralık for cc_neg, fitted vs actual
ax = axes[1, 1]
setup_ax(ax)
try:
    import statsmodels.formula.api as smf
    df_ols = agg.copy()
    event_ym_sb = pd.Period('2013-12', 'M')
    df_ols['post'] = (df_ols['ym'] >= event_ym_sb).astype(int)
    df_ols['time_idx'] = range(len(df_ols))
    df_ols2 = df_ols.rename(columns={'cc_neg_share': 'y'})
    mod = smf.ols('y ~ post + time_idx', data=df_ols2).fit(cov_type='HC3')
    beta_post = mod.params['post']
    pval_post = mod.pvalues['post']
    fitted = mod.fittedvalues.values
    ax.plot(xs_ts, agg['cc_neg_share'].values, color=COLORS_CAT['cc_neg'], lw=1.8, label='Actual cc_neg_share')
    ax.plot(xs_ts[:len(fitted)], fitted, color='black', lw=1.5, ls='--', label='OLS fitted')
    ev_ts = pd.Period('2013-12', 'M').to_timestamp()
    ax.axvline(ev_ts, color='#C0392B', lw=1.5, ls='--', alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(ev_ts, ylim[0] + (ylim[1]-ylim[0])*0.85, '17-25\nAralık', fontsize=8, color='#C0392B', ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    stars = sig_stars(pval_post)
    ax.text(0.05, 0.92, f'β_post = {beta_post:.4f}\np = {pval_post:.3f} {stars}\nHC3 SE',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)
except Exception as e:
    ax.text(0.5, 0.5, f'OLS error:\n{e}', transform=ax.transAxes, ha='center')
ax.set_title('D: Structural Break at 17-25 Aralık\nOLS fitted vs actual cc_neg_share', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')

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
fig2.suptitle('Term 24 — Political Dynamics Analysis (2011–2015)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Çözüm süreci — sec_neg_share HDP vs MHP vs AKP
ax = axes2[0, 0]
setup_ax(ax)
parties_sec = {
    'Adalet ve Kalkınma Partisi':   ('#E63329', 'AKP'),
    'Milliyetçi Hareket Partisi':   ('#8B0000', 'MHP'),
    'Halkların Demokratik Partisi': ('#6A0DAD', 'HDP'),
}
for party, (col, lbl) in parties_sec.items():
    pdata = monthly[monthly['party']==party].sort_values('ym')
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['sec_neg_share'].values, color=col, lw=1.8, label=lbl)
cozum_ts = pd.Period('2013-03', 'M').to_timestamp()
ax.axvline(cozum_ts, color='#27AE60', lw=1.5, ls='--')
ylim = ax.get_ylim()
ax.text(cozum_ts, ylim[0] + (ylim[1]-ylim[0])*0.88, 'Çözüm\nSüreci', fontsize=8, color='#27AE60', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
ax.set_title('A: Çözüm Süreci — Security Conflict Framing\nHDP vs MHP vs AKP sec_neg_share', fontsize=10, fontweight='bold')
ax.set_ylabel('sec_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: AKP security_neg monthly (framing Gezi as security threat)
ax = axes2[0, 1]
setup_ax(ax)
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
xs_akp = [p.to_timestamp() for p in akp_data['ym'].values]
ax.plot(xs_akp, akp_data['sec_neg_share'].values, color='#E63329', lw=1.8)
ax.fill_between(xs_akp, akp_data['sec_neg_share'].values, alpha=0.2, color='#E63329')
gezi_ts = pd.Period('2013-05', 'M').to_timestamp()
ax.axvline(gezi_ts, color='#2980B9', lw=1.5, ls='--')
ylim = ax.get_ylim()
ax.text(gezi_ts, ylim[0] + (ylim[1]-ylim[0])*0.88, 'Gezi', fontsize=8, color='#2980B9', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
ax.set_title('B: AKP Security Framing — Gezi as Security Threat\nAKP sec_neg_share monthly', fontsize=10, fontweight='bold')
ax.set_ylabel('AKP sec_neg_share')
ax.tick_params(axis='x', rotation=30)

# Panel C: 17-25 Aralık → hukuk devleti: cc_neg by party Dec 2013 - Mar 2014 zoomed
ax = axes2[1, 0]
setup_ax(ax)
zoom_start = pd.Period('2013-09', 'M')
zoom_end   = pd.Period('2014-06', 'M')
for party, (col, lbl) in {
    'Adalet ve Kalkınma Partisi':   ('#E63329', 'AKP'),
    'Cumhuriyet Halk Partisi':      ('#E87722', 'CHP'),
    'Halkların Demokratik Partisi': ('#6A0DAD', 'HDP'),
    'Milliyetçi Hareket Partisi':   ('#8B0000', 'MHP'),
}.items():
    pdata = monthly[monthly['party']==party].sort_values('ym')
    pdata = pdata[(pdata['ym'] >= zoom_start) & (pdata['ym'] <= zoom_end)]
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['cc_neg_share'].values, color=col, lw=1.8, marker='o', ms=4, label=lbl)
aralik_ts = pd.Period('2013-12', 'M').to_timestamp()
ax.axvline(aralik_ts, color='#C0392B', lw=1.5, ls='--')
ylim = ax.get_ylim()
ax.text(aralik_ts, ylim[0] + (ylim[1]-ylim[0])*0.88, '17-25\nAralık', fontsize=8, color='#C0392B', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('C: 17-25 Aralık → Hukuk Devleti Crisis\ncc_neg by party (Sep 2013 – Jun 2014)', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel D: Democratic stress index monthly with Gezi + 17-25 Aralık
ax = axes2[1, 1]
setup_ax(ax)
stress_agg = monthly.groupby('ym').agg(cc_neg=('cc_neg','sum'), dr_pos=('dr_pos','sum')).reset_index()
stress_agg['demo_stress'] = stress_agg['cc_neg'] / (stress_agg['dr_pos'] + 0.01)
stress_agg = stress_agg.sort_values('ym')
xs_s = [p.to_timestamp() for p in stress_agg['ym'].values]
ax.fill_between(xs_s, stress_agg['demo_stress'].values, alpha=0.25, color='#C0392B')
ax.plot(xs_s, stress_agg['demo_stress'].values, color='#C0392B', lw=1.8)
for ym_str, lbl, col in [('2013-05', 'Gezi', '#2980B9'), ('2013-12', '17-25\nAralık', '#C0392B')]:
    ev_ts = pd.Period(ym_str, 'M').to_timestamp()
    ax.axvline(ev_ts, color=col, lw=1.5, ls='--', alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(ev_ts, ylim[0] + (ylim[1]-ylim[0])*0.88, lbl, fontsize=8, color=col, ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
ax.set_title('D: Democratic Stress Index (cc_neg/dr_pos)\nwith Gezi and 17-25 Aralık', fontsize=10, fontweight='bold')
ax.set_ylabel('Demo Stress Index')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = f'{FIG_DIR}/term_{TERM}_political_dynamics.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('Term 24 done.')
