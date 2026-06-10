"""
Term 23 Deep Analysis — Causal Identification & Political Dynamics
Events: AKP kapatma davası (2008-07), Küresel Kriz (2008-09), Anayasa Referandumu (2010-09)
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
TERM    = 23
CSV     = f'{BASE}/Term_{TERM}/CSVs/term_{TERM}_beme_concept_edges.csv'
FIG_DIR = f'{BASE}/Term_{TERM}/Figures'

# ── colours ──────────────────────────────────────────────────────────────────
C = {
    'Adalet ve Kalkınma Partisi':  '#E63329',
    'Cumhuriyet Halk Partisi':     '#E87722',
    'Milliyetçi Hareket Partisi':  '#8B0000',
    'cc_neg':  '#C0392B',
    'eco_neg': '#2980B9',
    'dr_pos':  '#27AE60',
    'sec_neg': '#8E44AD',
}
BG, PBG = '#FAFAFA', '#F5F5F5'

# ── helper functions ──────────────────────────────────────────────────────────
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

def mann_kendall(x):
    x = np.array(x)
    n = len(x)
    if n < 4: return np.nan, np.nan
    s = sum(np.sign(x[j]-x[i]) for i in range(n) for j in range(i+1,n))
    v = n*(n-1)*(2*n+5)/18
    z = (s - np.sign(s))/np.sqrt(v) if v > 0 else 0
    p = 2*(1-stats.norm.cdf(abs(z)))
    return z, p

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
    # find closest x
    diffs = [abs((t - ev).days) for t in xs_ts]
    idx = int(np.argmin(diffs))
    ax.axvline(xs_ts[idx], color=color, lw=1.4, ls='--', alpha=0.8)
    ylim = ax.get_ylim()
    ypos = ylim[0] + (ylim[1]-ylim[0])*ymax
    ax.text(xs_ts[idx], ypos, label, fontsize=7, color=color, ha='center',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
monthly = build_monthly_panel(df)
agg = agg_all_parties(monthly)

xs_ts = [p.to_timestamp() for p in agg['ym'].values]

events = {
    'kapatma': ('2008-07', 'AKP\nKapatma', '#C0392B'),
    'kriz':    ('2008-09', 'Küresel\nKriz',   '#2980B9'),
    'ref':     ('2010-09', 'Anayasa\nRef.',    '#27AE60'),
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — causal_identification
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Term 23 — Causal Identification Analysis (2007–2011)', fontsize=14, fontweight='bold', y=0.98)

# Panel A: Monthly cc_neg_share + eco_neg_share with event markers
ax = axes[0, 0]
setup_ax(ax)
ax.plot(xs_ts, agg['cc_neg_share'].values, color=C['cc_neg'], lw=1.8, label='cc_neg_share')
ax.plot(xs_ts, agg['eco_neg_share'].values, color=C['eco_neg'], lw=1.8, label='eco_neg_share', ls='--')
for key, (ym, lbl, col) in events.items():
    add_event_vline(ax, xs_ts, ym, lbl, color=col)
ax.set_title('A: Constitutional vs Economic Negative Framing', fontsize=10, fontweight='bold')
ax.set_ylabel('Share of total mentions')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: Event study ±4 months around kapatma davası (cc_neg)
ax = axes[0, 1]
setup_ax(ax)
rel, norm = event_study(agg, '2008-07', 'cc_neg_share', window=4)
if rel is not None:
    ax.bar(rel, norm, color=[C['cc_neg'] if r >= 0 else '#AAB7B8' for r in rel],
           edgecolor='white', width=0.7)
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.axhline(1, color='gray', lw=1, ls=':')
    for r, n in zip(rel, norm):
        ax.text(r, n + 0.03, f'{n:.2f}', ha='center', fontsize=7)
ax.set_title('B: Event Study — AKP Kapatma Davası\n(cc_neg normalized to pre-event)', fontsize=10, fontweight='bold')
ax.set_xlabel('Months relative to event (Jul 2008)')
ax.set_ylabel('Normalized cc_neg_share')

# Panel C: Structural break OLS at Anayasa Ref (2010-09)
ax = axes[1, 0]
setup_ax(ax)
try:
    mod = structural_break_ols(agg, '2010-09', 'cc_neg_share')
    beta_post = mod.params['post']
    pval_post = mod.pvalues['post']
    ci = mod.conf_int().loc['post']
    ax.bar(['post coef'], [beta_post], color=C['cc_neg'] if beta_post > 0 else '#2980B9',
           edgecolor='black', width=0.4)
    ax.errorbar(['post coef'], [beta_post],
                yerr=[[beta_post - ci[0]], [ci[1] - beta_post]],
                fmt='none', color='black', capsize=6, lw=2)
    ax.axhline(0, color='black', lw=1)
    stars = sig_stars(pval_post)
    ax.text(0, beta_post + (ci[1]-beta_post)*1.1,
            f'β={beta_post:.4f}\np={pval_post:.3f} {stars}',
            ha='center', fontsize=9, fontweight='bold')
    ax.set_title('C: Structural Break OLS — Anayasa Referandumu (Sep 2010)\ncc_neg_share post coefficient (HC3)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Coefficient estimate')
except Exception as e:
    ax.text(0.5, 0.5, f'OLS error:\n{e}', transform=ax.transAxes, ha='center')

# Panel D: Frame displacement scatter eco_neg vs cc_neg
ax = axes[1, 1]
setup_ax(ax)
x = agg['eco_neg_share'].values
y = agg['cc_neg_share'].values
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]
r, pval = stats.pearsonr(x, y)
ax.scatter(x, y, color=C['cc_neg'], alpha=0.6, s=40, edgecolors='white', lw=0.5)
m, b = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, m*xline + b, color='black', lw=1.5, ls='--')
stars = sig_stars(pval)
ax.text(0.05, 0.92, f'Pearson r = {r:.3f}\np = {pval:.3f} {stars}',
        transform=ax.transAxes, fontsize=9, fontweight='bold',
        bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
ax.set_xlabel('eco_neg_share')
ax.set_ylabel('cc_neg_share')
ax.set_title('D: Frame Displacement Test\nH: Economic crisis → eco_neg surge → cc_neg displacement (r < 0?)', fontsize=10, fontweight='bold')
ax.text(0.5, 0.02, 'r < 0 = frames compete; r > 0 = frames co-occur',
        transform=ax.transAxes, fontsize=8, ha='center', color='gray')

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
fig2.suptitle('Term 23 — Political Dynamics Analysis (2007–2011)', fontsize=14, fontweight='bold', y=0.98)

parties_plot = {
    'Adalet ve Kalkınma Partisi': ('#E63329', 'AKP'),
    'Cumhuriyet Halk Partisi':    ('#E87722', 'CHP'),
    'Milliyetçi Hareket Partisi': ('#8B0000', 'MHP'),
}

# Panel A: cc_neg_share trajectory AKP vs CHP vs MHP
ax = axes2[0, 0]
setup_ax(ax)
for party, (col, lbl) in parties_plot.items():
    pdata = monthly[monthly['party']==party].sort_values('ym')
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['cc_neg_share'].values, color=col, lw=1.8, label=lbl)
for key, (ym, lbl, col) in events.items():
    add_event_vline(ax, xs_ts, ym, lbl, color=col, ymax=0.9)
ax.set_title('A: cc_neg_share Trajectory by Party', fontsize=10, fontweight='bold')
ax.set_ylabel('cc_neg_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel B: dr_pos_share trajectory AKP vs opposition
ax = axes2[0, 1]
setup_ax(ax)
for party, (col, lbl) in parties_plot.items():
    pdata = monthly[monthly['party']==party].sort_values('ym')
    if len(pdata) == 0: continue
    xs_p = [p.to_timestamp() for p in pdata['ym'].values]
    ax.plot(xs_p, pdata['dr_pos_share'].values, color=col, lw=1.8, label=lbl)
for key, (ym, lbl, col) in events.items():
    add_event_vline(ax, xs_ts, ym, lbl, color=col, ymax=0.9)
ax.set_title('B: Democratic Reform Positive Rhetoric (dr_pos_share)', fontsize=10, fontweight='bold')
ax.set_ylabel('dr_pos_share')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# Panel C: Democratic stress index monthly all parties
ax = axes2[1, 0]
setup_ax(ax)
stress_agg = monthly.groupby('ym').agg(
    cc_neg=('cc_neg','sum'), dr_pos=('dr_pos','sum')
).reset_index()
stress_agg['demo_stress'] = stress_agg['cc_neg'] / (stress_agg['dr_pos'] + 0.01)
stress_agg = stress_agg.sort_values('ym')
xs_s = [p.to_timestamp() for p in stress_agg['ym'].values]
ax.fill_between(xs_s, stress_agg['demo_stress'].values, alpha=0.3, color='#C0392B')
ax.plot(xs_s, stress_agg['demo_stress'].values, color='#C0392B', lw=1.8)
for key, (ym, lbl, col) in events.items():
    add_event_vline(ax, xs_s, ym, lbl, color=col, ymax=0.9)
ax.set_title('C: Democratic Stress Index (cc_neg / dr_pos)\nAll parties monthly', fontsize=10, fontweight='bold')
ax.set_ylabel('Demo Stress Index')
ax.tick_params(axis='x', rotation=30)

# Panel D: Pre vs Post kapatma AKP cc_neg mean bars with 95% CI
ax = axes2[1, 1]
setup_ax(ax)
t_stat, pval, pre_mean, post_mean = party_prepost(monthly, '2008-07', 'Adalet ve Kalkınma Partisi', 'cc_neg_share')
event_ym = pd.Period('2008-07', 'M')
akp_data = monthly[monthly['party']=='Adalet ve Kalkınma Partisi'].sort_values('ym')
pre_vals  = akp_data[akp_data['ym'] < event_ym]['cc_neg_share'].values
post_vals = akp_data[akp_data['ym'] >= event_ym]['cc_neg_share'].values

def ci95(x):
    if len(x) < 2: return 0
    return 1.96 * x.std() / np.sqrt(len(x))

bars = ax.bar(['Pre-Kapatma', 'Post-Kapatma'],
              [pre_mean, post_mean],
              color=['#AAB7B8', '#E63329'],
              edgecolor='black', width=0.5)
ax.errorbar(['Pre-Kapatma', 'Post-Kapatma'],
            [pre_mean, post_mean],
            yerr=[ci95(pre_vals), ci95(post_vals)],
            fmt='none', color='black', capsize=8, lw=2)
stars = sig_stars(pval)
ymax_bar = max(pre_mean + ci95(pre_vals), post_mean + ci95(post_vals))
ax.text(0.5, ymax_bar * 1.08, f't={t_stat:.2f}, p={pval:.3f} {stars}',
        ha='center', fontsize=9, fontweight='bold',
        transform=ax.get_xaxis_transform() if False else ax.transData)
ax.annotate(f't={t_stat:.2f}\np={pval:.3f} {stars}',
            xy=(0.5, ymax_bar*1.05), xycoords='data',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=2))
ax.set_title('D: AKP cc_neg_share — Pre vs Post Kapatma Davası\nwith 95% CI', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean cc_neg_share')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = f'{FIG_DIR}/term_{TERM}_political_dynamics.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Saved: {out2}')
print('Term 23 done.')
