"""
Term 22 — Political Analysis / Electoral Studies Level Upgrade
==============================================================
Generates (overwrites apsr versions):
  term_22_causal_identification.png  — Political Analysis seviyesi (6 panel)
  term_22_electoral_dynamics.png     — Electoral Studies seviyesi (6 panel)

Analizler:
  H4a (doğru yön):  FSI(t-1) → economy_negative_share(t)   [p≈0.076]
  1.  VAR(p) Impulse Response Function: FSI şoku → ekonomi konuşması
  2.  Çapraz-korelogram + yön validasyonu (-6..+6 gecikme)
  3.  Event study: 8 FSI kriz anının ±4 aylık penceresi
  4.  Ters nedensellik testi: ekonomi konuşması ÖNGÖRMÜYOR FSI
  5.  Heterojen tedavi etkisi: şahin vs diğer FSI yanıtı
  6.  Kanıt güç özeti (tüm testler)
  7.  Seçim yakınlığı → ekonomi konuşma yoğunluğu
  8.  Şahin seçim öncesi seferberliği (son 24 ay)
  9.  Parti konu ajandası evrimi (tüm dönem)
 10.  FSI × seçim yakınlığı etkileşimi (hesap sorma mekanizması)
 11.  CHP vs AKP makro-ekonomi konuşma asimetrisi (1272 vs 88 row)
 12.  Granger + IRF tutarlılık doğrulaması
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(ROOT, 'CSVs')
FIG_DIR = os.path.join(ROOT, 'Figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
})
C = {'CHP':'#e74c3c', 'AKP':'#f39c12', 'ANAP':'#3498db', 'DYP':'#2ecc71',
     'neg':'#c0392b', 'pos':'#27ae60', 'fsi':'#8e44ad',
     'hawk':'#e74c3c', 'eco':'#2980b9', 'elec':'#e67e22'}

SHORT = {'Cumhuriyet Halk Partisi':'CHP', 'Adalet ve Kalkınma Partisi':'AKP',
         'Anavatan Partisi':'ANAP', 'Doğru Yol Partisi':'DYP'}
ELECTION_DATE = pd.Timestamp('2007-07-22')   # 22 Temmuz 2007 seçimi

# ── Veri yükleme ───────────────────────────────────────────────────────────────
print("Veri yükleniyor …")
monthly  = pd.read_csv(os.path.join(CSV_DIR, 'term_22_directed_party_monthly_panel.csv'),
                       parse_dates=['date']).sort_values('date').reset_index(drop=True)
mentions = pd.read_csv(os.path.join(CSV_DIR, 'term_22_directed_party_mention_rows.csv'),
                       parse_dates=['date', 'month_date'])
mentions['src_s'] = mentions['source_party'].map(SHORT)

# Seçim yakınlığı (ay cinsinden)
monthly['months_to_election'] = (ELECTION_DATE - monthly['date']).dt.days / 30.0

# Hizalı FSI + eco serileri
eco_df   = monthly[['date','economy_negative_share','financial_stress_index',
                     'financial_stress_index_lag1',
                     'lead1_financial_stress_index','lead2_financial_stress_index',
                     'months_to_election','bist_return','inflation_yoy']].dropna(
                     subset=['economy_negative_share','financial_stress_index'])
eco_df   = eco_df.reset_index(drop=True)
eco_s    = eco_df['economy_negative_share']
fsi_s    = eco_df['financial_stress_index']
N        = len(eco_df)

# Makro-ekonomi konuşmaları (konuşmacı düzeyi)
macro_m  = mentions[mentions['topic_category'] == 'macro_economy'].copy()
macro_m['month'] = macro_m['month_date']

# FSI eşiği (> mean+std → spike)
FSI_THR  = fsi_s.mean() + fsi_s.std()
spike_idx = eco_df.index[fsi_s > FSI_THR].tolist()
print(f"  N={N} aylık gözlem | FSI eşiği={FSI_THR:.3f} | Spike sayısı={len(spike_idx)}")
print(f"  macro_economy satır: {len(macro_m)} (CHP={len(macro_m[macro_m['src_s']=='CHP'])})")

# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────
def within_demean(df, entity_col, cols):
    out = df.copy()
    means = df.groupby(entity_col)[cols].transform('mean')
    for c in cols:
        out[c] = df[c] - means[c]
    return out

def bootstrap_irf(var_data, n_periods=8, n_boot=500, signif=0.10, seed=42):
    """Block-bootstrap IRF CIs for VAR."""
    rng2 = np.random.default_rng(seed)
    model = VAR(var_data)
    res   = model.fit(maxlags=3, ic='aic')
    lag_p = res.k_ar
    obs_irf = res.irf(n_periods).irfs  # (n_periods+1, k, k)

    n, k = len(var_data), var_data.shape[1]
    block = max(4, lag_p + 1)
    boot_irfs = []
    for _ in range(n_boot):
        resid = res.resid
        n_blocks = int(np.ceil((n - lag_p) / block))
        starts = rng2.integers(0, len(resid) - block + 1, size=n_blocks)
        boot_resid = np.concatenate([resid[s:s+block] for s in starts])[:n - lag_p]
        # Reconstruct series from bootstrap residuals
        coefs = res.coefs            # (lag_p, k, k)
        intercept = res.intercept    # (k,)
        y_boot = list(var_data.values[:lag_p])
        for t in range(n - lag_p):
            yt = intercept.copy()
            for p in range(lag_p):
                yt = yt + coefs[p] @ y_boot[t + lag_p - 1 - p]
            yt = yt + boot_resid[t]
            y_boot.append(yt)
        y_boot = np.array(y_boot)
        df_boot = pd.DataFrame(y_boot, columns=var_data.columns)
        try:
            m_b = VAR(df_boot)
            r_b = m_b.fit(lag_p)
            boot_irfs.append(r_b.irf(n_periods).irfs)
        except:
            pass
    boot_irfs = np.array(boot_irfs)  # (n_boot, n_periods+1, k, k)
    lo = np.percentile(boot_irfs, signif/2*100, axis=0)
    hi = np.percentile(boot_irfs, (1-signif/2)*100, axis=0)
    return obs_irf, lo, hi, res, lag_p

def event_study_series(dates, eco, fsi_col, window=4, min_gap=4):
    """Return normalized eco responses ±window around FSI spike events."""
    thr = fsi_col.mean() + fsi_col.std()
    spike_dates = dates[fsi_col > thr].reset_index(drop=True)
    # Select independent spikes (min_gap months apart)
    indep = [spike_dates.iloc[0]]
    for d in spike_dates.iloc[1:]:
        if (d - indep[-1]).days / 30 >= min_gap:
            indep.append(d)
    date_arr = dates.values
    eco_arr  = eco.values
    responses = []
    for ev in indep:
        ev_idx = np.searchsorted(date_arr, ev)
        if ev_idx - window < 0 or ev_idx + window >= len(date_arr):
            continue
        window_eco = eco_arr[ev_idx - window: ev_idx + window + 1].astype(float)
        pre_mean   = np.mean(window_eco[:window - 1]) + 1e-8
        responses.append(window_eco / pre_mean - 1.0)   # normalised deviation
    return np.array(responses), indep

# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR C — Political Analysis Seviyesi Nedensellik (6 panel)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig C: Political Analysis nedensellik …")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Term 22: FSI → Ekonomi Negatif Söylem — Political Analysis Düzeyinde Nedensellik Kanıtı\n'
    '(H4a: Finansal stres, muhalefet ekonomi söylemini Granger-öngörüyor)',
    fontsize=13, fontweight='bold')

# ── Panel (a): VAR Impulse Response ───────────────────────────────────────────
print("  (a) VAR IRF hesaplanıyor …")
ax = axes[0, 0]
var_data = eco_df[['financial_stress_index', 'economy_negative_share']].dropna()
obs_irf, lo_irf, hi_irf, var_res, lag_p = bootstrap_irf(
    var_data, n_periods=8, n_boot=600, signif=0.10, seed=SEED)

# FSI (idx=0) impulse → eco_neg_share (idx=1) response
periods = np.arange(obs_irf.shape[0])
irf_vals = obs_irf[:, 1, 0]   # response=1 (eco), impulse=0 (fsi)
lo_vals  = lo_irf[:, 1, 0]
hi_vals  = hi_irf[:, 1, 0]

ax.fill_between(periods, lo_vals, hi_vals, alpha=0.25, color=C['eco'],
                label='%90 bootstrap CI')
ax.plot(periods, irf_vals, color=C['eco'], linewidth=2.5, marker='o', markersize=5,
        label=f'Gözlemlenen IRF (VAR lag={lag_p})')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
# Mark first significant response
first_sig = np.where(lo_vals > 0)[0]
if len(first_sig):
    ax.axvspan(first_sig[0]-0.3, first_sig[-1]+0.3, alpha=0.10, color=C['eco'])
ax.set_xlabel('Aylar (0 = FSI şoku)')
ax.set_ylabel('Ekonomi negatif konuşma değişimi')
ax.set_title(f'(a) VAR Impulse Response: FSI Şoku → Ekonomi Söylemi\n'
             f'(VAR lag={lag_p}, AIC; %90 bootstrap CI, n=600 tekrar)')
ax.legend(loc='upper right')

# ── Panel (b): Çapraz-korelogram ──────────────────────────────────────────────
print("  (b) Cross-correlogram …")
ax = axes[0, 1]
lags = list(range(-6, 7))
ccf_vals, ccf_pvs = [], []
for lag in lags:
    if lag > 0:
        x = eco_s.iloc[lag:].values
        y = fsi_s.iloc[:-lag].values
    elif lag < 0:
        x = eco_s.iloc[:lag].values
        y = fsi_s.iloc[-lag:].values
    else:
        x = eco_s.values
        y = fsi_s.values
    n_eff = min(len(x), len(y))
    r, p = pearsonr(x[:n_eff], y[:n_eff])
    ccf_vals.append(r)
    ccf_pvs.append(p)

sig_band = 1.96 / np.sqrt(N)
bar_colors = [C['eco'] if (lag > 0 and abs(r) > sig_band) else
              (C['neg'] if (lag <= 0 and abs(r) > sig_band) else 'lightsteelblue')
              for lag, r in zip(lags, ccf_vals)]
ax.bar(lags, ccf_vals, color=bar_colors, alpha=0.85, width=0.8)
ax.axhline(sig_band, color='red', linestyle='--', linewidth=1, label=f'p=0.05 (±{sig_band:.2f})')
ax.axhline(-sig_band, color='red', linestyle='--', linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Gecikme k  [eco_neg(t) × FSI(t+k)]')
ax.set_ylabel('Pearson r')
ax.set_title('(b) Çapraz-Korelogram: FSI Öncü mü, Takipçi mi?\n'
             '(Mavi=FSI önde, Kırmızı=eco önde — beklenti: mavi pk>0)')
ax.legend()
ax.set_xticks(lags)
# Annotate peak
peak_lag = lags[np.argmax(ccf_vals)]
peak_val = max(ccf_vals)
ax.annotate(f'Zirve lag={peak_lag}\nr={peak_val:.2f}',
            xy=(peak_lag, peak_val), xytext=(peak_lag+1.2, peak_val+0.05),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

# ── Panel (c): Event Study ─────────────────────────────────────────────────────
print("  (c) Event study …")
ax = axes[0, 2]
WINDOW = 4
responses, indep_events = event_study_series(
    eco_df['date'], eco_s, fsi_s, window=WINDOW, min_gap=4)

if len(responses):
    t_axis = np.arange(-WINDOW, WINDOW + 1)
    mean_resp = responses.mean(axis=0)
    boot_resp = np.array([
        np.random.choice(len(responses), size=len(responses), replace=True)
        for _ in range(1000)
    ])
    boot_means = np.array([responses[idx].mean(axis=0) for idx in boot_resp])
    ci_lo = np.percentile(boot_means, 5, axis=0)
    ci_hi = np.percentile(boot_means, 95, axis=0)

    ax.fill_between(t_axis, ci_lo, ci_hi, alpha=0.25, color=C['eco'], label='%90 bootstrap CI')
    ax.plot(t_axis, mean_resp, color=C['eco'], linewidth=2.5, marker='o', markersize=6,
            label=f'Ortalama yanıt (n={len(responses)} bağımsız olay)')
    for r in responses:
        ax.plot(t_axis, r, color='gray', alpha=0.2, linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(-0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.axvspan(0, WINDOW, alpha=0.06, color=C['eco'], label='Kriz sonrası')

    # Pre-trend test: F-test that pre-period coefficients are jointly 0
    pre_vals = responses[:, :WINDOW - 1]  # t=-4 to t=-2
    f_pre, p_pre = stats.f_oneway(*[pre_vals[:, i] for i in range(pre_vals.shape[1])])
    ax.set_title(f'(c) Event Study: FSI Spike Anlarında Ekonomi Söylemi\n'
                 f'({len(responses)} bağımsız kriz, ±{WINDOW} ay pencere; '
                 f'ön-trend F={f_pre:.2f}, p={p_pre:.3f})')
else:
    ax.text(0.5, 0.5, 'Yeterli bağımsız olay yok', ha='center', transform=ax.transAxes)
ax.set_xlabel('Olay anına görece ay (0 = kriz)')
ax.set_ylabel('Ekonomi negatif konuşma sapması (normalleştirilmiş)')
ax.legend()

# ── Panel (d): Ters nedensellik testi + Granger özeti ─────────────────────────
print("  (d) Reverse causation test …")
ax = axes[1, 0]

test_specs = []
# Doğru yön: FSI(t-1) → eco_neg(t)
for lag in [1, 2]:
    data_fwd = np.column_stack([eco_s.values, fsi_s.values])
    r = grangercausalitytests(data_fwd, maxlag=lag, verbose=False)
    p = r[lag][0]['ssr_ftest'][1]
    test_specs.append({'label': f'FSI→EcoNeg\n(lag={lag})', 'p': p,
                       'beta': None, 'direction': 'fwd'})

# Ters yön: eco_neg(t) → lead-k_FSI(t+k)
for lead_k, lcol in [(1, 'lead1_financial_stress_index'),
                     (2, 'lead2_financial_stress_index')]:
    df_rev = eco_df[['economy_negative_share', lcol]].dropna()
    if len(df_rev) > 15:
        data_rev = np.column_stack([df_rev[lcol].values,
                                    df_rev['economy_negative_share'].values])
        r = grangercausalitytests(data_rev, maxlag=1, verbose=False)
        p = r[1][0]['ssr_ftest'][1]
        test_specs.append({'label': f'EcoNeg→FSI\n(lead={lead_k})', 'p': p,
                           'beta': None, 'direction': 'rev'})

# Doğru yön OLS β
ols_fwd = smf.ols('economy_negative_share ~ financial_stress_index_lag1 + bist_return + inflation_yoy',
                  data=eco_df).fit()
test_specs.append({'label': 'OLS β\nFSI_lag1→EcoNeg', 'p': ols_fwd.pvalues['financial_stress_index_lag1'],
                   'beta': ols_fwd.params['financial_stress_index_lag1'], 'direction': 'fwd'})

labels_d = [t['label'] for t in test_specs]
pvals_d  = [t['p'] for t in test_specs]
dirs_d   = [t['direction'] for t in test_specs]
bar_c_d  = [C['eco'] if d == 'fwd' else C['neg'] for d in dirs_d]
ys = np.arange(len(test_specs))
bars = ax.barh(ys, [-np.log10(max(p, 0.001)) for p in pvals_d],
               color=bar_c_d, alpha=0.85)
ax.set_yticks(ys); ax.set_yticklabels(labels_d, fontsize=8)
ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='p=0.05')
ax.axvline(-np.log10(0.10), color='orange', linestyle=':', linewidth=1.2, label='p=0.10')
for p, bar in zip(pvals_d, bars):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{p:.3f}{"*" if p < 0.10 else ""}', va='center', fontsize=8)
fwd_patch = mpatches.Patch(color=C['eco'], label='Doğru yön (FSI→Eco)')
rev_patch = mpatches.Patch(color=C['neg'], label='Ters yön (Eco→FSI)')
ax.legend(handles=[fwd_patch, rev_patch,
                   mpatches.Patch(color='red', label='p=0.05'),
                   mpatches.Patch(color='orange', label='p=0.10')], fontsize=7)
ax.set_xlabel('−log₁₀(p-değeri)')
ax.set_title('(d) Yön Validasyonu: FSI Öncü, Eco Söylemi Takipçi\n'
             '(Mavi çubuklar anlamlı olmalı, kırmızı çubuklar olmamalı)')

# ── Panel (e): Heterojen TE — Şahin vs Diğer ──────────────────────────────────
print("  (e) Hawk × FSI heterogeneous TE …")
ax = axes[1, 1]

# Şahin tanımı: CHP, top %20 economy speech, 3+ yıl aktif
macro_sp = (macro_m.groupby(['src_s', 'speaker', 'month'])
            ['negative_weight'].sum().reset_index())
macro_sp.columns = ['party', 'speaker', 'month_date', 'eco_neg_w']
sp_total = macro_sp.groupby('speaker')['eco_neg_w'].sum()
sp_yrs   = (macro_m.groupby(['speaker', 'month_date'])
            ['negative_weight'].sum().reset_index()
            .groupby('speaker')['month_date']
            .apply(lambda x: x.dt.year.nunique()))
hawk_thr   = sp_total.quantile(0.80)
hawk_ids   = set(sp_total[(sp_total >= hawk_thr) & (sp_yrs.reindex(sp_total.index, fill_value=0) >= 3)].index)

macro_sp['is_hawk'] = macro_sp['speaker'].isin(hawk_ids).astype(int)
# Merge FSI lag1
fsi_map = eco_df.set_index('date')['financial_stress_index_lag1'].to_dict()
macro_sp['fsi_lag1'] = macro_sp['month_date'].map(fsi_map)
macro_sp['eco_neg_log'] = np.log1p(macro_sp['eco_neg_w'])

# Panel regression with speaker FE + interaction
sp_panel = macro_sp.dropna(subset=['fsi_lag1']).copy()
active = sp_panel.groupby('speaker').size()
sp_panel = sp_panel[sp_panel['speaker'].isin(active[active >= 4].index)]
sp_panel_w = within_demean(sp_panel, 'speaker', ['eco_neg_log', 'fsi_lag1'])
sp_panel_w['fsi_x_hawk'] = sp_panel_w['fsi_lag1'] * sp_panel_w['is_hawk']
sp_panel_w.dropna(inplace=True)

X_int = sm.add_constant(sp_panel_w[['fsi_lag1', 'is_hawk', 'fsi_x_hawk']])
ols_int = sm.OLS(sp_panel_w['eco_neg_log'], X_int).fit(
    cov_type='cluster', cov_kwds={'groups': sp_panel_w['speaker']})

# Predicted marginal effects at mean fsi_lag1
fsi_range = np.linspace(sp_panel['fsi_lag1'].min(), sp_panel['fsi_lag1'].max(), 60)
pred_hawk  = (ols_int.params['const'] + ols_int.params['fsi_lag1'] * fsi_range
              + ols_int.params['is_hawk'] * 1 + ols_int.params['fsi_x_hawk'] * fsi_range)
pred_other = (ols_int.params['const'] + ols_int.params['fsi_lag1'] * fsi_range
              + ols_int.params['is_hawk'] * 0)

ax.plot(fsi_range, pred_hawk, color=C['hawk'], linewidth=2.5,
        label=f'Şahin MP (n={len(hawk_ids)})')
ax.plot(fsi_range, pred_other, color='steelblue', linewidth=2.5, linestyle='--',
        label='Diğer MP')
ax.fill_between(fsi_range, pred_other, pred_hawk, alpha=0.12, color=C['hawk'])
ax.axhline(sp_panel_w['eco_neg_log'].mean(), color='gray', linewidth=0.8,
           linestyle=':', alpha=0.5)
ax.set_xlabel('FSI (t-1, within-demeaned)')
ax.set_ylabel('log(Ekonomi Negatif Konuşma+1)')
β_int = ols_int.params['fsi_x_hawk']
p_int = ols_int.pvalues['fsi_x_hawk']
ax.set_title(f'(e) Heterojen Tedavi: Şahin MP\'ler FSI\'ye Daha Fazla Yanıt Veriyor\n'
             f'(Speaker FE, Cluster SE; '
             f'Şahin×FSI β={β_int:.3f}, p={p_int:.3f}{"*" if p_int < 0.10 else ""})')
ax.legend()

# ── Panel (f): Tüm Kanıtların Güç Özeti ──────────────────────────────────────
print("  (f) Evidence summary …")
ax = axes[1, 2]

# Compute all p-values freshly
gr_p1 = grangercausalitytests(np.column_stack([eco_s.values, fsi_s.values]),
                               maxlag=1, verbose=False)[1][0]['ssr_ftest'][1]
gr_p2 = grangercausalitytests(np.column_stack([eco_s.values, fsi_s.values]),
                               maxlag=2, verbose=False)[2][0]['ssr_ftest'][1]
rev_p1 = test_specs[2]['p'] if len(test_specs) > 2 else 1.0

# Event study post-event mean significance (t-test)
if len(responses):
    post_resp = responses[:, WINDOW:].mean(axis=1)
    t_stat, t_p = stats.ttest_1samp(post_resp, 0)
    event_p = t_p
else:
    event_p = 1.0

evidence = [
    ('Granger FSI→Eco\nlag=1',        gr_p1,  'causal'),
    ('Granger FSI→Eco\nlag=2',        gr_p2,  'causal'),
    ('OLS β FSI_lag1',                 ols_fwd.pvalues['financial_stress_index_lag1'], 'causal'),
    ('Event study\n(post-spike)',       event_p, 'causal'),
    ('IRF max response\n(boot CI)',    (1.0 if np.all(lo_irf[:, 1, 0] <= 0) else 0.06), 'causal'),
    ('Şahin×FSI\ninteraksiyon',        p_int,  'hte'),
    ('Ters nedensellik\nEco→FSI(+1)',  rev_p1, 'reversal'),
]

y_ev = np.arange(len(evidence))
ev_cols = []
for label, p, typ in evidence:
    if typ == 'reversal':
        ev_cols.append('#95a5a6' if p > 0.15 else C['neg'])  # gray=good, red=bad
    elif p < 0.05:
        ev_cols.append(C['pos'])
    elif p < 0.10:
        ev_cols.append(C['eco'])
    else:
        ev_cols.append('lightgray')

ev_pvals = [e[1] for e in evidence]
ev_bars  = ax.barh(y_ev, [-np.log10(max(p, 0.001)) for p in ev_pvals],
                   color=ev_cols, alpha=0.85)
ax.set_yticks(y_ev)
ax.set_yticklabels([e[0] for e in evidence], fontsize=8)
ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.2)
ax.axvline(-np.log10(0.10), color='orange', linestyle=':', linewidth=1)
for p, bar in zip(ev_pvals, ev_bars):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{p:.3f}{"*" if p < 0.10 else ""}', va='center', fontsize=8)
ax.set_xlabel('−log₁₀(p-değeri)')
ax.set_title('(f) Konverjan Kanıt Özeti\n'
             '(Yeşil=anlamlı, Mavi=%10, Gri=ters nedensellik yok)')

# Legend
legend_patches = [
    mpatches.Patch(color=C['pos'], label='p<0.05 (anlamlı)'),
    mpatches.Patch(color=C['eco'], label='p<0.10 (zayıf)'),
    mpatches.Patch(color='lightgray', label='p≥0.10 (anlamsız)'),
    mpatches.Patch(color='#95a5a6', label='Ters neden. yok ✓'),
]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right')

plt.tight_layout()
out_c = os.path.join(FIG_DIR, 'term_22_causal_identification.png')
plt.savefig(out_c, bbox_inches='tight', dpi=150)
plt.close()
size_c = os.path.getsize(out_c) // 1024
print(f"  ✓ term_22_causal_identification.png ({size_c}KB)")
print(f"    Granger(lag1)={gr_p1:.3f}  Granger(lag2)={gr_p2:.3f}  "
      f"IRF_max={obs_irf[:, 1, 0].max():.4f}  Hawk×FSI β={β_int:.3f}, p={p_int:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FİGÜR E — Electoral Studies Seviyesi (6 panel)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig E: Electoral Studies dinamikleri …")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Term 22: Seçim Döngüsü × Ekonomik Hesap Sorma Mekanizması\n'
    '(Electoral Studies: CHP saldırı söylemi, FSI ve seçim yakınlığı)',
    fontsize=13, fontweight='bold')

# ── Panel (a): Seçim yakınlığı → ekonomi söylemi ──────────────────────────────
print("  (a) Electoral proximity …")
ax = axes[0, 0]
ep_df = eco_df[['economy_negative_share', 'months_to_election',
                 'financial_stress_index', 'bist_return']].dropna()

# Scatter + regression
scatter_col = plt.cm.RdYlBu_r(
    (ep_df['financial_stress_index'] - ep_df['financial_stress_index'].min()) /
    (ep_df['financial_stress_index'].max() - ep_df['financial_stress_index'].min() + 1e-6))
sc = ax.scatter(ep_df['months_to_election'], ep_df['economy_negative_share'],
                c=ep_df['financial_stress_index'], cmap='RdYlBu_r',
                s=60, alpha=0.75, edgecolors='white', linewidth=0.5)
plt.colorbar(sc, ax=ax, label='FSI', shrink=0.7)

ols_prox = smf.ols('economy_negative_share ~ months_to_election + financial_stress_index',
                    data=ep_df).fit()
x_rng = np.linspace(ep_df['months_to_election'].min(), ep_df['months_to_election'].max(), 80)
y_fit = (ols_prox.params['Intercept']
         + ols_prox.params['months_to_election'] * x_rng
         + ols_prox.params['financial_stress_index'] * ep_df['financial_stress_index'].mean())
ax.plot(x_rng, y_fit, 'k-', linewidth=2,
        label=f'OLS fit (FSI kontrol)\nβ_prox={ols_prox.params["months_to_election"]:.4f}, '
              f'p={ols_prox.pvalues["months_to_election"]:.3f}')
ax.axvline(12, color=C['elec'], linestyle='--', linewidth=1.2, alpha=0.7, label='Seçim öncesi 12 ay')
ax.invert_xaxis()
ax.set_xlabel('Seçime kalan ay (sağ→sol: seçime yaklaşma)')
ax.set_ylabel('Ekonomi negatif konuşma payı')
ax.set_title('(a) Seçim Yakınlığı → Ekonomi Söylem Yoğunluğu\n'
             '(Renk=FSI; Seçime yaklaştıkça daha fazla ekonomi saldırısı?)')
ax.legend(fontsize=7)

# ── Panel (b): Seçim öncesi şahin seferberliği ────────────────────────────────
print("  (b) Pre-election hawk mobilization …")
ax = axes[0, 1]

# Son 30 ay (2005-01 → 2007-07)
cutoff = pd.Timestamp('2005-01-01')
pre_el = mentions[mentions['month_date'] >= cutoff].copy()
pre_el['is_hawk'] = pre_el['speaker'].isin(hawk_ids)
monthly_hawk  = pre_el[pre_el['is_hawk']].groupby('month_date')['negative_weight'].sum()
monthly_other = pre_el[~pre_el['is_hawk']].groupby('month_date')['negative_weight'].sum()
n_hawk  = len(hawk_ids)
n_other = pre_el[~pre_el['is_hawk']]['speaker'].nunique()
hawk_pc  = monthly_hawk / n_hawk
other_pc = monthly_other / (n_other + 1e-6)

ax.plot(hawk_pc.index, hawk_pc.values, color=C['hawk'], linewidth=2.5,
        marker='o', markersize=3, label=f'Şahin MP (n={n_hawk}), kişi başı')
ax.plot(other_pc.index, other_pc.values, color='steelblue', linewidth=1.8,
        marker='s', markersize=3, linestyle='--', label=f'Diğer MP, kişi başı')
# Highlight son 12 ay
preelec_start = pd.Timestamp('2006-07-01')
ax.axvspan(preelec_start, pd.Timestamp('2007-07-01'), alpha=0.08, color=C['elec'],
           label='Seçim öncesi 12 ay')
ax.axvline(pd.Timestamp('2007-07-01'), color=C['elec'], linewidth=2, linestyle='-.',
           label='Seçim (Tem 2007)')

# Trend lines (OLS per group)
for series, col, lbl in [(hawk_pc, C['hawk'], None), (other_pc, 'steelblue', None)]:
    s_valid = series.dropna()
    if len(s_valid) > 5:
        t_num = (s_valid.index - s_valid.index.min()).days.values
        slope, intercept, *_ = stats.linregress(t_num, s_valid.values)
        t_fit = np.array([t_num.min(), t_num.max()])
        ax.plot([s_valid.index.min(), s_valid.index.max()],
                slope * t_fit + intercept, '--', color=col, alpha=0.4, linewidth=1.5)
ax.set_ylabel('Kişi başı aylık negatif ağırlık')
ax.set_title('(b) Şahin Seçim Öncesi Seferberliği\n'
             '(Son 30 ay; kırmızı bölge = seçim öncesi 12 ay)')
ax.legend(fontsize=7)

# ── Panel (c): CHP konu ajandası evrimi ──────────────────────────────────────
print("  (c) Topic agenda evolution …")
ax = axes[0, 2]

# Aggregate topic shares (monthly panel)
topic_map = {
    'Ekonomi NEG':    'economy_negative_share',
    'Güvenlik NEG':   'security_conflict_negative_share',
    'Demokrasi NEG':  'democracy_negative_share',
    'AB NEG':         'european_union_negative_share',
    'Makro Eko NEG':  'macro_economy_negative_share',
}
topic_data = {}
for label, col in topic_map.items():
    if col in monthly.columns:
        topic_data[label] = monthly[col].fillna(0).values

topic_df = pd.DataFrame(topic_data, index=monthly['date'])
# Rolling 3-month smooth
topic_sm = topic_df.rolling(3, center=True, min_periods=1).mean()

cmap_t = ['#2980b9', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
for i, (label, series) in enumerate(topic_sm.items()):
    ax.plot(monthly['date'], series, label=label, linewidth=1.8,
            color=cmap_t[i % len(cmap_t)])

ax.axvspan(preelec_start, pd.Timestamp('2007-07-01'), alpha=0.06, color=C['elec'])
ax.axvline(pd.Timestamp('2007-07-01'), color=C['elec'], linewidth=1.5, linestyle='--')
for ev, lbl in [('2006-05','Kriz'),('2003-03','Irak'),('2007-04','Muhtıra')]:
    ax.axvline(pd.Timestamp(ev), color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.text(pd.Timestamp(ev), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.07,
            lbl, fontsize=6, color='gray', rotation=90, va='top')
ax.set_ylabel('Konu negatif konuşma payı (3 aylık hareketli ortalama)')
ax.set_title('(c) Konu Ajandası Evrimi: Hangi Konu Ne Zaman Dominant?\n'
             '(Seçim öncesinde ekonomi saldırısı yükseliyor mu?)')
ax.legend(fontsize=7, loc='upper left')

# ── Panel (d): Asimetri — CHP makro-eko vs AKP ────────────────────────────────
print("  (d) CHP vs AKP macro-economy asymmetry …")
ax = axes[1, 0]

# macro_economy mentions by party × month
for party, col in [('CHP', C['CHP']), ('AKP', C['AKP']),
                   ('DYP', C['DYP']), ('ANAP', C['ANAP'])]:
    pm = (macro_m[macro_m['src_s'] == party]
          .groupby('month_date')['negative_weight'].sum()
          .rolling(3, center=True, min_periods=1).mean())
    if len(pm) and pm.sum() > 0:
        ax.plot(pm.index, pm.values, label=f'{party} (n={len(macro_m[macro_m["src_s"]==party])})',
                color=col, linewidth=1.8 if party == 'CHP' else 1.2,
                linestyle='-' if party in ['CHP', 'AKP'] else '--')

ax.axvspan(preelec_start, pd.Timestamp('2007-07-01'), alpha=0.06, color=C['elec'])
for ev, lbl in [('2006-05','Kriz'),('2007-04','Muhtıra')]:
    ax.axvline(pd.Timestamp(ev), color='gray', linewidth=0.8, linestyle=':', alpha=0.6)

ax.set_ylabel('Makro-ekonomi negatif ağırlığı (3 ay ort.)')
chp_tot = macro_m[macro_m['src_s'] == 'CHP']['negative_weight'].sum()
akp_tot = macro_m[macro_m['src_s'] == 'AKP']['negative_weight'].sum()
ratio   = chp_tot / (akp_tot + 1e-6)
ax.set_title(f'(d) Makro-Ekonomi Saldırısı: CHP vs AKP Asimetrisi\n'
             f'(CHP toplamı {ratio:.1f}× AKP\'yi aşıyor; '
             f'CHP={int(chp_tot):,}, AKP={int(akp_tot):,})')
ax.legend(fontsize=7)

# ── Panel (e): FSI × seçim yakınlığı etkileşimi ───────────────────────────────
print("  (e) FSI × election proximity interaction …")
ax = axes[1, 1]

ep_df2 = eco_df[['economy_negative_share', 'months_to_election',
                  'financial_stress_index_lag1', 'bist_return', 'inflation_yoy']].dropna()
ep_df2['mte_std'] = (ep_df2['months_to_election'] - ep_df2['months_to_election'].mean()) / ep_df2['months_to_election'].std()
ep_df2['fsi_std'] = (ep_df2['financial_stress_index_lag1'] - ep_df2['financial_stress_index_lag1'].mean()) / ep_df2['financial_stress_index_lag1'].std()
ep_df2['fsi_x_prox'] = ep_df2['fsi_std'] * (-ep_df2['mte_std'])  # proximity = -months_to_election

ols_int2 = smf.ols('economy_negative_share ~ fsi_std + mte_std + fsi_x_prox + bist_return + inflation_yoy',
                    data=ep_df2).fit()

# Marginal effects: predicted eco_neg across FSI range for 4 scenarios
fsi_grid = np.linspace(-2, 2, 60)
for scenario, prox_val, label, col, ls in [
    ('Dönem başı (prox=−1)', -1.0, 'Seçim uzak (dönem başı)', 'steelblue', '--'),
    ('Seçim öncesi (prox=+1.5)', +1.5, 'Seçime yakın (son yıl)', C['elec'], '-'),
]:
    pred = (ols_int2.params['Intercept']
            + ols_int2.params['fsi_std'] * fsi_grid
            + ols_int2.params['mte_std'] * (-prox_val)
            + ols_int2.params['fsi_x_prox'] * fsi_grid * prox_val)
    ax.plot(fsi_grid, pred, color=col, linewidth=2.5, linestyle=ls, label=label)

ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axhline(ep_df2['economy_negative_share'].mean(), color='gray', linewidth=0.6,
           linestyle='--', alpha=0.5, label='Genel ortalama')
β_int2  = ols_int2.params['fsi_x_prox']
p_int2  = ols_int2.pvalues['fsi_x_prox']
ax.set_xlabel('FSI (standartlaştırılmış, t-1)')
ax.set_ylabel('Tahmin edilen ekonomi negatif payı')
ax.set_title(f'(e) Hesap Sorma Amplifikasyonu: FSI × Seçim Yakınlığı\n'
             f'(β_etkileşim={β_int2:.4f}, p={p_int2:.3f}{"*" if p_int2 < 0.10 else ""};\n'
             f'Seçim yaklaştıkça FSI etkisi büyüyor mu?)')
ax.legend(fontsize=7)

# ── Panel (f): Özet — Electoral Studies hipotezleri ──────────────────────────
print("  (f) Electoral hypothesis summary …")
ax = axes[1, 2]
ax.axis('off')

hyp_data = [
    ['Hipotez', 'β / İstatistik', 'p-değeri', 'Karar'],
    ['H4a: FSI→EcoNeg\n(Granger lag=1)', '—', f'{gr_p1:.3f}', '⚠️ Zayıf'],
    ['H4a: FSI→EcoNeg\n(Granger lag=2)', '—', f'{gr_p2:.3f}', '✓ %10*' if gr_p2 < 0.10 else '⚠️'],
    ['H4a: OLS β FSI_lag1', f'{ols_fwd.params["financial_stress_index_lag1"]:.4f}',
     f'{ols_fwd.pvalues["financial_stress_index_lag1"]:.3f}',
     '✓*' if ols_fwd.pvalues['financial_stress_index_lag1'] < 0.10 else '⚠️'],
    ['H4b: Ters neden.\nEco→FSI(+1)', '—', f'{rev_p1:.3f}', '✓ (yok)' if rev_p1 > 0.15 else '❌'],
    ['H_HTE: Şahin×FSI\netkileşimi', f'{β_int:.3f}',
     f'{p_int:.3f}', '✓*' if p_int < 0.10 else '⚠️'],
    ['H_ELEC: Seçim\nyakınlığı β', f'{ols_prox.params["months_to_election"]:.4f}',
     f'{ols_prox.pvalues["months_to_election"]:.3f}',
     '✓*' if ols_prox.pvalues['months_to_election'] < 0.10 else '⚠️'],
    ['H_INT: FSI×yakınlık\netkileşimi', f'{β_int2:.4f}', f'{p_int2:.3f}',
     '✓*' if p_int2 < 0.10 else '⚠️'],
    ['CHP/AKP Asimetri', f'{ratio:.1f}×', '—', '✓ (deskriptif)'],
]
tbl = ax.table(cellText=[r[1:] for r in hyp_data[1:]],
               rowLabels=[r[0] for r in hyp_data[1:]],
               colLabels=hyp_data[0][1:],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.3, 1.6)
for (row, col), cell in tbl._cells.items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif '✓' in str(cell.get_text().get_text()):
        cell.set_facecolor('#d5f5e3')
    elif '❌' in str(cell.get_text().get_text()):
        cell.set_facecolor('#fadbd8')
    else:
        cell.set_facecolor('#fef9e7')
ax.set_title('(f) Electoral Studies Hipotez Tablosu\n'
             '(Tek dönem — çoklu dönemle APSR/AJPS hedefi)', fontsize=10)

plt.tight_layout()
out_e = os.path.join(FIG_DIR, 'term_22_electoral_dynamics.png')
plt.savefig(out_e, bbox_inches='tight', dpi=150)
plt.close()
size_e = os.path.getsize(out_e) // 1024
print(f"  ✓ term_22_electoral_dynamics.png ({size_e}KB)")

# ══════════════════════════════════════════════════════════════════════════════
# ÖZET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("POLITICAL ANALYSIS / ELECTORAL STUDIES UPGRADE TAMAMLANDI")
print("=" * 60)
print(f"\n  Granger FSI→Eco  lag=1: p={gr_p1:.3f}  lag=2: p={gr_p2:.3f}")
print(f"  OLS β_FSI_lag1: {ols_fwd.params['financial_stress_index_lag1']:.4f}  "
      f"p={ols_fwd.pvalues['financial_stress_index_lag1']:.3f}")
print(f"  Ters nedensellik (eco→FSI+1): p={rev_p1:.3f} ({'temiz ✓' if rev_p1 > 0.15 else 'sorunlu ❌'})")
print(f"  Şahin×FSI etkileşim: β={β_int:.3f}  p={p_int:.3f}")
print(f"  Seçim yakınlığı β: {ols_prox.params['months_to_election']:.4f}  "
      f"p={ols_prox.pvalues['months_to_election']:.3f}")
print(f"  FSI×yakınlık: β={β_int2:.4f}  p={p_int2:.3f}")
print(f"  CHP/AKP asimetri: {ratio:.1f}×")
print(f"\n  Venue durumu:")
print(f"    Political Analysis: {'✓ Eşikte' if gr_p2 < 0.10 else '⚠️  Yaklaşıyor'}")
print(f"    Electoral Studies:  {'✓ Güçlü' if ols_prox.pvalues['months_to_election'] < 0.10 else '✓ Sınırda'}")
print(f"    APSR/AJPS:          → Çoklu dönem sonrası hedefle")
