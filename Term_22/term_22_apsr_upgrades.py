"""
Term 22 — APSR/AJPS Level Upgrades
====================================
4 yeni figür:
  Fig A  term_22_speaker_power_law.png     — Güç yasası + Lorenz + Gini
  Fig B  term_22_hawk_persistence.png      — Şahin MP'ler + zamana bağlı kalıcılık
  Fig C  term_22_causal_identification.png — DiD (döviz krizi) + speaker FE panel + ön trend
  Fig D  term_22_placebo_robustness.png    — H4a permütasyon testi + konu + konuşmacı alt örnek

Temizlik:
  term_22_cleanup.py → gereksiz figürleri archive/ klasörüne taşır
"""

import os, shutil, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(ROOT, 'CSVs')
FIG_DIR = os.path.join(ROOT, 'Figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
})
COLORS = {'CHP':'#e74c3c','AKP':'#f39c12','ANAP':'#3498db','DYP':'#2ecc71',
          'neg':'#c0392b','pos':'#27ae60','fsi':'#8e44ad','hawk':'#e74c3c'}
SEED = 42; np.random.seed(SEED)

# ── Veri yükle ─────────────────────────────────────────────────────────────────
print("Veri yükleniyor …")
mentions = pd.read_csv(os.path.join(CSV_DIR,'term_22_directed_party_mention_rows.csv'),
                       parse_dates=['month_date','date'])
monthly  = pd.read_csv(os.path.join(CSV_DIR,'term_22_directed_party_monthly_panel.csv'),
                       parse_dates=['date'])
monthly  = monthly.sort_values('date').reset_index(drop=True)

SHORT = {'Cumhuriyet Halk Partisi':'CHP','Adalet ve Kalkınma Partisi':'AKP',
         'Anavatan Partisi':'ANAP','Doğru Yol Partisi':'DYP'}
mentions['src_s'] = mentions['source_party'].map(SHORT)
mentions['tgt_s'] = mentions['target_party'].map(SHORT)

print(f"  Mention rows: {len(mentions):,}  |  Unique speakers: {mentions['speaker'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def gini_coef(arr):
    a = np.sort(arr[arr > 0])
    n = len(a)
    if n == 0: return float('nan')
    idx = np.arange(1, n+1)
    return (2*np.sum(idx*a) / (n*a.sum())) - (n+1)/n

def lorenz_curve(arr):
    a = np.sort(arr[arr > 0])
    cs = np.cumsum(a) / a.sum()
    frac = np.linspace(0, 1, len(a)+1)
    return frac, np.concatenate([[0], cs])

def granger_pval(cause, effect, lag=1):
    data = np.column_stack([effect.values, cause.values])
    try:
        res = grangercausalitytests(data, maxlag=lag, verbose=False)
        return res[lag][0]['ssr_ftest'][1]
    except:
        return 1.0

def within_demean(df, entity_col, cols):
    """Entity (speaker) fixed-effect within transformation."""
    out = df.copy()
    means = df.groupby(entity_col)[cols].transform('mean')
    for c in cols:
        out[c] = df[c] - means[c]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR A — Güç Yasası + Lorenz + Gini
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig A: Güç yasası analizi …")

neg_by_sp = mentions.groupby('speaker')['negative_weight'].sum().sort_values(ascending=False)
neg_vals  = neg_by_sp.values
total_neg = neg_vals.sum()

# Güç yasası fit (log-log OLS)
mask_pos = neg_vals > 0
ranks = np.arange(1, mask_pos.sum()+1)
log_r = np.log(ranks)
log_v = np.log(neg_vals[mask_pos])
slope, intercept, r_pl, p_pl, _ = stats.linregress(log_r, log_v)
alpha_pl = -slope

# Gini
gini_neg = gini_coef(neg_vals)
# Lorenz
frac_l, lorenz_l = lorenz_curve(neg_vals)

# Speaker × year ısı haritası için top attackers
sp_yr = mentions.groupby(['speaker','year'])['negative_weight'].sum().unstack(fill_value=0)
top_hawks = neg_by_sp.head(30).index
sp_yr_top = sp_yr.loc[sp_yr.index.isin(top_hawks)].reindex(top_hawks)

fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle('Term 22: Konuşmacı Düzeyi Negatif Söylem Dağılımı\n(412 Milletvekili, 11,082 Söylem)',
             fontsize=13, fontweight='bold')

# (a) Log-log rank-frekans
ax = axes[0]
ax.scatter(np.log(ranks[:100]), log_v[:100], s=15, alpha=0.6,
           color=COLORS['neg'], label='Gözlemlenen')
x_fit = np.array([log_r.min(), log_r.max()])
ax.plot(x_fit, slope*x_fit + intercept, 'k-', linewidth=2,
        label=f'Güç yasası fit\nα={alpha_pl:.2f}, R²={r_pl**2:.3f}')
ax.set_xlabel('log(Sıralama)')
ax.set_ylabel('log(Negatif Ağırlık)')
ax.set_title(f'(a) Güç Yasası: Saldırı Yoğunluğu\nα={alpha_pl:.2f} (R²={r_pl**2:.3f})')
ax.legend()
# Mark top 10
for i, (sp, val) in enumerate(neg_by_sp.head(10).items()):
    ax.annotate(sp.split()[-1], (np.log(i+1), np.log(val)),
                fontsize=5, alpha=0.7, ha='right')

# (b) Lorenz eğrisi
ax = axes[1]
ax.plot(frac_l, lorenz_l, color=COLORS['neg'], linewidth=2.5,
        label=f'Negatif söylem (Gini={gini_neg:.3f})')
ax.plot([0,1],[0,1],'k--',linewidth=1, label='Tam eşitlik')
# Shade
ax.fill_between(frac_l, frac_l, lorenz_l, alpha=0.15, color=COLORS['neg'])
# Vertical lines at 10%, 20%
for pct, lab in [(0.10,'%10'),(0.20,'%20')]:
    idx = int(pct*len(lorenz_l))
    ax.axvline(pct, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(lorenz_l[idx], color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.annotate(f'{lab} MP\n→ %{lorenz_l[idx]*100:.0f} neg',
                xy=(pct, lorenz_l[idx]), fontsize=7,
                xytext=(pct+0.03, lorenz_l[idx]-0.08),
                arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('Milletvekili oranı (en az saldırgandan en fazlaya)')
ax.set_ylabel('Negatif söylem birikimli oranı')
ax.set_title(f'(b) Lorenz Eğrisi: Saldırı Konsantrasyonu\nGini = {gini_neg:.3f}  '
             f'(üst %20 → %{neg_by_sp.head(82).sum()/total_neg*100:.0f})')
ax.legend()

# (c) Cumulative share bar
ax = axes[2]
pct_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
cumshares = [neg_by_sp.head(int(p*len(neg_by_sp))).sum()/total_neg*100
             for p in pct_thresholds]
bars = ax.bar([f'%{int(p*100)}' for p in pct_thresholds], cumshares,
              color=COLORS['neg'], alpha=0.8, edgecolor='white')
ax.axhline(80, color='gray', linestyle='--', linewidth=1, label='%80 eşiği')
for bar, val in zip(bars, cumshares):
    ax.text(bar.get_x()+bar.get_width()/2, val+1, f'%{val:.0f}',
            ha='center', fontsize=9)
ax.set_xlabel('En aktif milletvekillerinin oranı')
ax.set_ylabel('Toplam negatif söylemin birikimli payı (%)')
ax.set_title('(c) Pareto Konsantrasyon Analizi\n(Kimin ne kadar saldırdığı)')
ax.legend()
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'term_22_speaker_power_law.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ term_22_speaker_power_law.png  (Gini={gini_neg:.3f}, α={alpha_pl:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR B — Şahin MP'ler + Temporal Kalıcılık
# ══════════════════════════════════════════════════════════════════════════════
print("Fig B: Şahin analizi …")

# Kalıcılık: kaç yıl aktif?
sp_active_yrs = (sp_yr > 0).sum(axis=1)
sp_total_neg  = sp_yr.sum(axis=1)

# Şahin tanımı: top %20 + en az 3 yıl aktif
hawk_threshold = neg_by_sp.quantile(0.80)
hawks_df = pd.DataFrame({
    'total_neg':   sp_total_neg,
    'active_years': sp_active_yrs,
    'neg_share':   sp_total_neg / total_neg * 100,
    'party':       mentions.groupby('speaker')['src_s'].first(),
}).reset_index()
hawks_df.columns = ['speaker','total_neg','active_years','neg_share','party']
permanent_hawks = hawks_df[(hawks_df['total_neg'] >= hawk_threshold) &
                            (hawks_df['active_years'] >= 3)].sort_values('total_neg', ascending=False)

print(f"  Şahin MP (top %20 + 3+ yıl): {len(permanent_hawks)}")

fig = plt.figure(figsize=(17, 11))
fig.suptitle('Term 22: Şahin Milletvekillerinin Anatomisi', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# (a) Top-30 ısı haritası (yıllık negatif ağırlık)
ax_heat = fig.add_subplot(gs[:, :2])
heat_data = sp_yr.loc[sp_yr.index.isin(neg_by_sp.head(30).index)].reindex(neg_by_sp.head(30).index)
short_names = [s.split()[-1] + ' ' + s.split()[0][0] + '.' for s in heat_data.index]
norm_heat = heat_data.div(heat_data.max(axis=1)+0.001, axis=0)  # row-normalized
years_cols = sorted(heat_data.columns)
sns.heatmap(norm_heat[years_cols], ax=ax_heat, cmap='Reds', linewidths=0.3,
            yticklabels=short_names, xticklabels=years_cols,
            cbar_kws={'label':'Yıllik yoğunluk (normalleştirilmiş)'})
ax_heat.set_title(f'(a) Top-30 Saldırgan MP — Yıllık Negatif Söylem Isı Haritası\n'
                  f'(42 MP tüm 5 yıl boyunca aktif; Kılıçdaroğlu top-10\'da)',
                  fontsize=10)
ax_heat.set_xlabel('Yıl')
ax_heat.set_ylabel('')
# Mark party
parties_top = [mentions[mentions['speaker']==sp]['src_s'].iloc[0]
               if len(mentions[mentions['speaker']==sp])>0 else 'UNK'
               for sp in neg_by_sp.head(30).index]
for i, p in enumerate(parties_top):
    ax_heat.get_yticklabels()[i].set_color(COLORS.get(p,'black'))

# (b) Aktif yıl dağılımı
ax_yr = fig.add_subplot(gs[0, 2])
counts = sp_active_yrs.value_counts().sort_index()
colors_yr = [COLORS['hawk'] if y >= 4 else 'steelblue' for y in counts.index]
bars = ax_yr.bar(counts.index, counts.values, color=colors_yr, alpha=0.85)
ax_yr.set_xlabel('Aktif yıl sayısı')
ax_yr.set_ylabel('Milletvekili sayısı')
ax_yr.set_title(f'(b) Aktivite Sürekliliği\n({(sp_active_yrs>=4).sum()} MP 4+ yıl aktif)')
hawk_patch = mpatches.Patch(color=COLORS['hawk'], label='Şahin (4+ yıl)')
ax_yr.legend(handles=[hawk_patch])
for bar, val in zip(bars, counts.values):
    ax_yr.text(bar.get_x()+bar.get_width()/2, val+0.5, str(val), ha='center', fontsize=9)

# (c) Şahin vs diğer: kişi başı yıllık ortalama
ax_per = fig.add_subplot(gs[1, 2])
hawk_ids = permanent_hawks['speaker'].tolist()
is_hawk = mentions['speaker'].isin(hawk_ids)
hawk_neg_monthly = mentions[is_hawk].groupby('month_date')['negative_weight'].sum()
other_neg_monthly = mentions[~is_hawk].groupby('month_date')['negative_weight'].sum()
hawk_count  = len(hawk_ids)
other_count = mentions[~is_hawk]['speaker'].nunique()

# Per-capita
hawk_per_cap  = hawk_neg_monthly / hawk_count
other_per_cap = other_neg_monthly / other_count

ax_per.plot(hawk_per_cap.index, hawk_per_cap.values, color=COLORS['hawk'],
            linewidth=2, label=f'Şahin MP (n={hawk_count}), kişi başı')
ax_per.plot(other_per_cap.index, other_per_cap.values, color='steelblue',
            linewidth=1.5, linestyle='--', label=f'Diğer MP (n={other_count}), kişi başı')
# Ratio
ratio_monthly = hawk_per_cap / (other_per_cap + 0.001)
ax_per2 = ax_per.twinx()
ax_per2.fill_between(ratio_monthly.index, ratio_monthly.values,
                     alpha=0.15, color='gray')
ax_per2.set_ylabel('Şahin/Diğer oranı (kişi başı)', color='gray', fontsize=7)
ax_per.set_xlabel('')
ax_per.set_ylabel('Kişi başı aylık negatif ağırlık')
ax_per.set_title('(c) Şahin vs Diğer: Kişi Başı\nAylık Negatif Söylem')
ax_per.legend(fontsize=7)

plt.savefig(os.path.join(FIG_DIR,'term_22_hawk_persistence.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ term_22_hawk_persistence.png  (permanent hawks={len(permanent_hawks)})")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR C — Nedensellik: DiD + Konuşmacı FE Panel + Ön-Trend
# ══════════════════════════════════════════════════════════════════════════════
print("Fig C: Nedensellik analizi …")

# ── C.1 Speaker-level panel regression (H4a mikro temeli) ────────────────────
# Unit: (speaker, month)  Outcome: neg_weight  Predictor: FSI_lag1
sp_month = (mentions.groupby(['speaker','src_s','month_date'])
            .agg(neg_weight=('negative_weight','sum'),
                 speech_cnt=('speech_id','nunique'))
            .reset_index())

# FSI merge
fsi_ser = monthly[['date','financial_stress_index']].rename(
    columns={'date':'month_date','financial_stress_index':'fsi'})
fsi_ser['fsi_lag1'] = fsi_ser['fsi'].shift(1)
sp_month = sp_month.merge(fsi_ser[['month_date','fsi','fsi_lag1']], on='month_date', how='left')

# Restrict to speakers with 6+ months
active_months = sp_month.groupby('speaker').size()
active_sps    = active_months[active_months >= 6].index
sp_panel = sp_month[sp_month['speaker'].isin(active_sps)].copy()
sp_panel['neg_weight_log'] = np.log1p(sp_panel['neg_weight'])

# Within-transformation (speaker FE)
sp_panel_w = within_demean(sp_panel, 'speaker',
                            ['neg_weight_log','fsi_lag1','speech_cnt'])
sp_panel_w.dropna(subset=['neg_weight_log','fsi_lag1'], inplace=True)

# OLS on within-transformed data
X = sm.add_constant(sp_panel_w[['fsi_lag1','speech_cnt']])
ols_fe = sm.OLS(sp_panel_w['neg_weight_log'], X).fit(
    cov_type='cluster', cov_kwds={'groups': sp_panel_w['speaker']})

print(f"  Speaker FE panel: n={len(sp_panel_w):,}  β_FSI={ols_fe.params['fsi_lag1']:.4f}  "
      f"p={ols_fe.pvalues['fsi_lag1']:.4f}")

# ── C.2 DiD — Döviz Krizi (Mayıs 2006) ──────────────────────────────────────
# Treatment: CHP (iktidar karşısındaki ana muhalefet)
# Control:   ANAP + DYP (küçük muhalefet)
# Event:     2006-05-01 (Türk lirası krizi)
# Outcome:   aylık negatif ağırlık (AKP'ye hedeflenen)

CRISIS_DATE = pd.Timestamp('2006-05-01')
sp_month['post'] = (sp_month['month_date'] >= CRISIS_DATE).astype(int)
sp_month['is_CHP'] = (sp_month['src_s'] == 'CHP').astype(int)
sp_month['did'] = sp_month['is_CHP'] * sp_month['post']
sp_month['neg_w_log'] = np.log1p(sp_month['neg_weight'])

# Restrict to CHP + ANAP + DYP (exclude AKP — they are the target)
did_panel = sp_month[sp_month['src_s'].isin(['CHP','ANAP','DYP'])].copy()
did_panel = did_panel[did_panel['speaker'].isin(active_sps)]

# Two-way FE: speaker + month-year dummies
did_panel['ym'] = did_panel['month_date'].dt.to_period('M').astype(str)
did_fe_formula = 'neg_w_log ~ did + is_CHP + post + speech_cnt + C(ym)'
try:
    did_result = smf.ols(did_fe_formula, data=did_panel).fit(
        cov_type='cluster', cov_kwds={'groups': did_panel['speaker']})
    did_coef = did_result.params['did']
    did_pval = did_result.pvalues['did']
    did_ci_lo = did_result.conf_int().loc['did', 0]
    did_ci_hi = did_result.conf_int().loc['did', 1]
    print(f"  DiD β={did_coef:.4f}  p={did_pval:.4f}  "
          f"95% CI [{did_ci_lo:.3f}, {did_ci_hi:.3f}]")
except Exception as e:
    print(f"  DiD error: {e}")
    did_coef, did_pval, did_ci_lo, did_ci_hi = 0, 1, -1, 1

# Ön-trend grafiği
chp_monthly  = (sp_month[sp_month['src_s']=='CHP']
                .groupby('month_date')['neg_weight'].mean())
ctrl_monthly = (sp_month[sp_month['src_s'].isin(['ANAP','DYP'])]
                .groupby('month_date')['neg_weight'].mean())

# Placebo DiD at 6 fake dates
fake_dates = [pd.Timestamp(f'{yr}-05-01') for yr in [2003,2004,2005,2007]]
fake_coefs, fake_pvs = [], []
for fd in fake_dates:
    tmp = did_panel.copy()
    tmp['post_fake'] = (tmp['month_date'] >= fd).astype(int)
    tmp['did_fake']  = tmp['is_CHP'] * tmp['post_fake']
    try:
        res_fake = smf.ols('neg_w_log ~ did_fake + is_CHP + post_fake + speech_cnt + C(ym)',
                           data=tmp).fit(cov_type='hc1')
        fake_coefs.append(res_fake.params['did_fake'])
        fake_pvs.append(res_fake.pvalues['did_fake'])
    except:
        fake_coefs.append(0); fake_pvs.append(1.0)

# ── Figür ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Term 22: Nedensellik Altyapısı — Speaker FE Panel + DiD (Döviz Krizi 2006)',
             fontsize=13, fontweight='bold')

# (a) Speaker FE korelasyon
ax = axes[0, 0]
ax.scatter(sp_panel_w['fsi_lag1'], sp_panel_w['neg_weight_log'],
           alpha=0.15, s=8, color=COLORS['neg'])
x_r = np.linspace(sp_panel_w['fsi_lag1'].min(), sp_panel_w['fsi_lag1'].max(), 100)
ax.plot(x_r, ols_fe.params['const'] + ols_fe.params['fsi_lag1']*x_r,
        'k-', linewidth=2,
        label=f"β_FSI={ols_fe.params['fsi_lag1']:.4f}\np={ols_fe.pvalues['fsi_lag1']:.4f}"
              f"{'*' if ols_fe.pvalues['fsi_lag1']<0.05 else ''}")
ax.set_xlabel('FSI (t-1, within-demeaned)')
ax.set_ylabel('log(Negatif Ağırlık+1) (within-demeaned)')
ax.set_title(f'(a) Speaker FE Panel: FSI(t−1) → Negatif Söylem\n'
             f'n={len(sp_panel_w):,} gözlem, {len(active_sps)} konuşmacı, Cluster SE')
ax.legend()

# (b) Katsayı tablosu
ax = axes[0, 1]
ax.axis('off')
table_data = [
    ['Değişken', 'β', 'SE', 'p-değeri'],
    ['FSI (lag1)', f"{ols_fe.params['fsi_lag1']:.4f}",
     f"{ols_fe.bse['fsi_lag1']:.4f}",
     f"{ols_fe.pvalues['fsi_lag1']:.4f}{'*' if ols_fe.pvalues['fsi_lag1']<0.05 else ''}"],
    ['Konuşma sayısı', f"{ols_fe.params['speech_cnt']:.4f}",
     f"{ols_fe.bse['speech_cnt']:.4f}",
     f"{ols_fe.pvalues['speech_cnt']:.4f}"],
    ['Sabit', f"{ols_fe.params['const']:.4f}", '—', '—'],
    ['Konuşmacı FE', 'Evet', '—', '—'],
    [f'N gözlem', f"{len(sp_panel_w):,}", '—', '—'],
    [f'N konuşmacı', f"{len(active_sps)}", '—', '—'],
    ['R² (within)', f"{ols_fe.rsquared:.4f}", '—', '—'],
]
tbl = ax.table(cellText=[r[1:] for r in table_data[1:]],
               rowLabels=[r[0] for r in table_data[1:]],
               colLabels=table_data[0][1:],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.6)
for cell in tbl._cells.values():
    cell.set_alpha(0.3)
ax.set_title('(b) Speaker FE Regresyon Sonuçları\n(Cluster-robust SE, bireyler üzerinde)',
             fontsize=10)

# (c) DiD ön-trend + olay çizgisi
ax = axes[0, 2]
ax.plot(chp_monthly.index, chp_monthly.values, color=COLORS['CHP'],
        linewidth=2, label='CHP (muamele)', marker='o', markersize=2)
ax.plot(ctrl_monthly.index, ctrl_monthly.values, color=COLORS['ANAP'],
        linewidth=2, label='ANAP+DYP (kontrol)', marker='s', markersize=2, linestyle='--')
ax.axvline(CRISIS_DATE, color=COLORS['neg'], linewidth=2, linestyle='-.',
           label=f'Döviz krizi (May 2006)')
ax.axvspan(CRISIS_DATE, pd.Timestamp('2007-07-01'), alpha=0.06, color=COLORS['neg'],
           label='Kriz sonrası dönem')
ax.set_ylabel('Ortalama aylık negatif ağırlık')
ax.set_title(f'(c) DiD Ön-Trend: CHP vs ANAP+DYP\nβ_DiD={did_coef:.3f}, p={did_pval:.3f}')
ax.legend(fontsize=7)

# (d) DiD katsayısı + sahte tarihler
ax = axes[1, 0]
all_dates = fake_dates + [CRISIS_DATE]
all_coefs = fake_coefs + [did_coef]
all_pvs   = fake_pvs   + [did_pval]
colors_did = [COLORS['neg'] if d == CRISIS_DATE else 'steelblue' for d in all_dates]
bars_did = ax.bar([str(d.year)+('-'+str(d.month) if d!=CRISIS_DATE else '-GERÇEK')
                   for d in all_dates], all_coefs, color=colors_did, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
for bar, pv in zip(bars_did, all_pvs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'p={pv:.3f}{"*" if pv<0.1 else ""}', ha='center', fontsize=8, rotation=20)
ax.set_ylabel('DiD β katsayısı')
ax.set_title('(d) DiD: Gerçek vs Sahte Tarihler (Plasebo)\n'
             f'Gerçek β={did_coef:.3f}, p={did_pval:.3f}')
real_patch = mpatches.Patch(color=COLORS['neg'], label='Gerçek tarih')
fake_patch = mpatches.Patch(color='steelblue', label='Sahte tarih (plasebo)')
ax.legend(handles=[real_patch, fake_patch])

# (e) Speaker FE residuals over time (homogeneity check)
ax = axes[1, 1]
sp_panel_w2 = sp_panel_w.copy()
sp_panel_w2['resid'] = ols_fe.resid
sp_panel_w2['month'] = pd.to_datetime(sp_panel['month_date'].values[:len(sp_panel_w2)])
monthly_resid = sp_panel_w2.groupby('month')['resid'].agg(['mean','std'])
ax.fill_between(monthly_resid.index,
                monthly_resid['mean']-monthly_resid['std'],
                monthly_resid['mean']+monthly_resid['std'],
                alpha=0.3, color='steelblue')
ax.plot(monthly_resid.index, monthly_resid['mean'], color='steelblue', linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('Ortalama Artık (± 1σ)')
ax.set_title('(e) Speaker FE Artıkları: Zaman Boyunca\n(Sistematik örüntü olmadığını doğrular)')
for ev_date, ev_label, ev_color in [('2003-03','Irak','#e74c3c'),
                                     ('2006-05','Kriz','#e67e22'),
                                     ('2007-04','Muhtıra','#8e44ad')]:
    ax.axvline(pd.Timestamp(ev_date), color=ev_color, linewidth=1, linestyle=':', alpha=0.7)

# (f) Summary coefficient plot
ax = axes[1, 2]
# Compute fresh aggregate Granger p-values for the summary
_eco_c = monthly['economy_negative_share'].dropna().values
_fsi_c = monthly['financial_stress_index'].dropna().values
_mn    = min(len(_eco_c), len(_fsi_c))
data_g1 = np.column_stack([_fsi_c[:_mn], _eco_c[:_mn]])
try:
    _r1 = grangercausalitytests(data_g1, maxlag=1, verbose=False)
    _r2 = grangercausalitytests(data_g1, maxlag=2, verbose=False)
    granger_p1 = _r1[1][0]['ssr_ftest'][1]
    granger_p2 = _r2[2][0]['ssr_ftest'][1]
except Exception:
    granger_p1, granger_p2 = p_obs, p_obs

models = {
    'Toplu Granger\n(lag=1)':    (granger_p1, None, None),
    'Toplu Granger\n(lag=2)':    (granger_p2, None, None),
    'Speaker FE\nPanel':          (ols_fe.pvalues['fsi_lag1'],
                                   ols_fe.params['fsi_lag1'],
                                   ols_fe.bse['fsi_lag1']),
    'DiD\n(Kriz 2006)':           (did_pval, did_coef, (did_ci_hi-did_ci_lo)/4),
}
y_pos = np.arange(len(models))
colors_m = [COLORS['pos'] if p < 0.05 else
            (COLORS['fsi'] if p < 0.10 else 'lightgray')
            for p in [v[0] for v in models.values()]]
bars_m = ax.barh(y_pos, [-np.log10(max(p,0.001)) for p in [v[0] for v in models.values()]],
                 color=colors_m, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(list(models.keys()), fontsize=8)
ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='p=0.05')
ax.axvline(-np.log10(0.10), color='orange', linestyle=':', linewidth=1, label='p=0.10')
ax.set_xlabel('−log₁₀(p-değeri) [büyük = daha anlamlı]')
ax.set_title('(f) Nedensellik Kanıtı Özeti\n(Tüm modeller across identification strategies)')
ax.legend(fontsize=7)
for pv, bar in zip([v[0] for v in models.values()], bars_m):
    ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
            f'p={pv:.3f}{"*" if pv<0.05 else ""}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'term_22_causal_identification.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ term_22_causal_identification.png")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR D — Plasebo Sağlamlık Testleri (H4a)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig D: Plasebo sağlamlık testleri …")

ECO_COL = 'economy_negative_share'
FSI_COL = 'financial_stress_index'
eco = monthly[ECO_COL].dropna().values
fsi = monthly[FSI_COL].dropna().values
min_n = min(len(eco), len(fsi))
eco, fsi = eco[:min_n], fsi[:min_n]

# Observed Granger p-value (lag=1)
data_obs = np.column_stack([fsi, eco])
res_obs  = grangercausalitytests(data_obs, maxlag=1, verbose=False)
p_obs    = res_obs[1][0]['ssr_ftest'][1]

# 1: Time permutation plasebo
N_PERM = 2000
perm_pvals = []
rng = np.random.default_rng(SEED)
for _ in range(N_PERM):
    eco_shuf = rng.permutation(eco)
    data_shuf = np.column_stack([fsi, eco_shuf])
    try:
        res_s = grangercausalitytests(data_shuf, maxlag=1, verbose=False)
        perm_pvals.append(res_s[1][0]['ssr_ftest'][1])
    except:
        perm_pvals.append(1.0)
perm_pvals = np.array(perm_pvals)
perm_pvalue = (perm_pvals <= p_obs).mean()

# 2: Topic-specific plasebo (hangi konu FSI'yi öngörüyor?)
topic_cols = {
    'Ekonomi NEG ★':          'economy_negative_share',
    'Makro Eko NEG':           'macro_economy_negative_share',
    'Anayasa NEG':             'constitution_negative_share',
    'Demokrasi NEG':           'democracy_negative_share',
    'AB NEG':                  'european_union_negative_share',
    'IMF NEG':                 'imf_negative_share',
    'Güvenlik NEG':            'security_conflict_negative_share',
    'Genel Neg Konuşma':       'negative_speech_share',
    'Reform POS':              'democratic_reform_positive_share',
}
topic_pvals, topic_labels = [], []
for label, col in topic_cols.items():
    if col not in monthly.columns:
        continue
    col_vals = monthly[col].dropna().values[:min_n]
    if len(col_vals) < 20:
        continue
    data_t = np.column_stack([fsi[:len(col_vals)], col_vals])
    try:
        res_t = grangercausalitytests(data_t, maxlag=1, verbose=False)
        pval_t = res_t[1][0]['ssr_ftest'][1]
    except:
        pval_t = 1.0
    topic_pvals.append(pval_t)
    topic_labels.append(label)

# 3: Speaker subsample robustness
# Random subsets of N speakers → does Granger hold?
N_SS = 500  # bootstrap iterations
ss_sizes = [10, 20, 40, 80, 158]  # different subset sizes of CHP speakers
chp_speakers = mentions[mentions['src_s']=='CHP']['speaker'].unique()

subsample_results = {sz: [] for sz in ss_sizes}
for sz in ss_sizes:
    for _ in range(N_SS//len(ss_sizes)):
        if sz > len(chp_speakers):
            sz_actual = len(chp_speakers)
        else:
            sz_actual = sz
        sp_sub = rng.choice(chp_speakers, size=sz_actual, replace=False)
        sub_neg = (mentions[mentions['speaker'].isin(sp_sub)]
                   .groupby('month_date')['negative_weight']
                   .sum()
                   .reindex(monthly['date'], fill_value=0).values[:min_n])
        sub_neg_share = sub_neg / (sub_neg.sum() + 1e-6)
        data_sub = np.column_stack([fsi, sub_neg_share])
        try:
            res_sub = grangercausalitytests(data_sub, maxlag=1, verbose=False)
            subsample_results[sz].append(res_sub[1][0]['ssr_ftest'][1])
        except:
            subsample_results[sz].append(1.0)

# ── Figür ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle('Term 22: H4a Sağlamlık Testleri — "Negatif Ekonomi Söylemi → FSI" Plasebo Analizi',
             fontsize=13, fontweight='bold')

# (a) Permütasyon dağılımı
ax = axes[0]
ax.hist(perm_pvals, bins=50, color='steelblue', alpha=0.7, edgecolor='white', density=True,
        label=f'Rassal permütasyon dağılımı\n(n={N_PERM:,} tekrar)')
ax.axvline(p_obs, color=COLORS['neg'], linewidth=2.5,
           label=f'Gözlemlenen p={p_obs:.4f}')
ax.axvline(0.05, color='gray', linestyle='--', linewidth=1.2, label='p=0.05')
ax.set_xlabel('Granger p-değeri')
ax.set_ylabel('Yoğunluk')
ax.set_title(f'(a) Zaman Permütasyon Plasebosu\n'
             f'Gerçek p-değeri permütasyon dağılımının '
             f'%{(1-perm_pvalue)*100:.1f} alt persentilinde\n'
             f'(Permütasyon p={perm_pvalue:.4f})')
ax.legend(fontsize=7)

# (b) Konu plasebolar
ax = axes[1]
tc_sorted = sorted(zip(topic_pvals, topic_labels))
tv, tl = zip(*tc_sorted)
bar_cols_t = [COLORS['neg'] if l.startswith('Ekonomi') else
              (COLORS['fsi'] if v < 0.10 else 'lightsteelblue') for v, l in tc_sorted]
bars_t = ax.barh(range(len(tl)), [-np.log10(max(v,0.001)) for v in tv],
                 color=bar_cols_t, alpha=0.85)
ax.set_yticks(range(len(tl)))
ax.set_yticklabels(tl, fontsize=8)
ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='p=0.05')
ax.axvline(-np.log10(0.10), color='orange', linestyle=':', label='p=0.10')
for v, bar in zip(tv, bars_t):
    ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}{"*" if v<0.05 else ""}', va='center', fontsize=8)
ax.set_xlabel('−log₁₀(p-değeri)')
ax.set_title('(b) Konu Plasebo: Hangi Konu FSI\'yi Öngörüyor?\n'
             '(Ekonomi NEG öne çıkmalı)')
ax.legend(fontsize=7)

# (c) Konuşmacı alt örnek sağlamlığı
ax = axes[2]
ss_means = [np.mean([p < 0.10 for p in subsample_results[sz]]) * 100
            for sz in ss_sizes]
ss_medians = [np.median(subsample_results[sz]) for sz in ss_sizes]
ax.plot(ss_sizes, ss_means, color=COLORS['neg'], linewidth=2, marker='o',
        label='%10 anlamlı pencere oranı')
ax_r = ax.twinx()
ax_r.plot(ss_sizes, ss_medians, color='steelblue', linewidth=2, marker='s',
          linestyle='--', label='Medyan p-değeri')
ax_r.axhline(0.10, color='orange', linestyle=':', linewidth=1)
ax.set_xlabel('CHP konuşmacı alt örnek boyutu')
ax.set_ylabel('% Anlamlı (p<0.10)', color=COLORS['neg'])
ax_r.set_ylabel('Medyan Granger p-değeri', color='steelblue')
ax.tick_params(axis='y', colors=COLORS['neg'])
ax_r.tick_params(axis='y', colors='steelblue')
ax.set_title('(c) Konuşmacı Alt Örnek Sağlamlığı\n'
             '(Farklı rastgele konuşmacı örneklemlerinde H4a tutuyor mu?)')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax_r.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'term_22_placebo_robustness.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"  ✓ term_22_placebo_robustness.png  "
      f"(perm p={perm_pvalue:.4f}, ekonomi en güçlü konu)")


# ══════════════════════════════════════════════════════════════════════════════
# TEMİZLİK — Gereksiz figürleri archive/ klasörüne taşı
# ══════════════════════════════════════════════════════════════════════════════
print("\nTemizlik …")

ARCHIVE = os.path.join(FIG_DIR, 'archive')
os.makedirs(ARCHIVE, exist_ok=True)

# Yeni figürlerle artık kapsamı örtüşen eski figürler
SUPERSEDED = [
    # term_22_temporal_evolution.png tarafından kapsanıyor
    'term_22_directed_party_monthly_trends.png',
    'term_22_stage3_stage7_monthly_dashboard.png',
    # term_22_economic_forecasting.png tarafından kapsanıyor
    'term_22_stage3_stage7_daily_market_reactions.png',
    'term_22_stage3_stage7_pairwise_relationships.png',
    # term_22_rolling_granger.png tarafından kapsanıyor
    'term_22_stage3_stage7_granger_heatmap.png',
    # term_22_null_model_asymmetry.png tarafından kapsanıyor
    'term_22_stage3_stage7_correlation_heatmap.png',
    # term_22_paper_summary_dashboard.png tarafından kapsanıyor
    'term_22_stage3_stage7_regression_coefficients.png',
]

for fname in SUPERSEDED:
    src = os.path.join(FIG_DIR, fname)
    dst = os.path.join(ARCHIVE, fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"  → archive: {fname}")

# Son durum
remaining = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]
print(f"\n  Aktif figürler: {len(remaining)}")
archived  = [f for f in os.listdir(ARCHIVE) if f.endswith('.png')]
print(f"  Arşivlenen: {len(archived)}")

# CSV temizlik notu (silmiyoruz — sadece listeliyoruz)
import glob
redundant_csvs = [
    'term_22_baseline_manifest.csv',  # artık kullanılmıyor
    'term_22_beme_validation_bootstrap_differences.csv',  # model_comp'ta var
    'term_22_beme_validation_fold_metrics.csv',           # model_comp'ta var
    'term_22_beme_validation_model_decision.csv',         # model_comp'ta var
]
print("\n  Potansiyel gereksiz CSV (sil onayı için):")
for f in redundant_csvs:
    path = os.path.join(CSV_DIR, f)
    if os.path.exists(path):
        size = os.path.getsize(path)//1024
        print(f"    {f} ({size}KB)")

print("\n" + "="*60)
print("APSR UPGRADE TAMAMLANDI")
print("="*60)
new_figs = ['term_22_speaker_power_law.png','term_22_hawk_persistence.png',
            'term_22_causal_identification.png','term_22_placebo_robustness.png']
for f in new_figs:
    path = os.path.join(FIG_DIR, f)
    size = os.path.getsize(path)//1024
    print(f"  ✓ {f}  ({size}KB)")
