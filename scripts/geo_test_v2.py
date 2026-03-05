import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIG ──────────────────────────────────────────────────────────────────

CSV_PATH              = '/Users/kate/geo-experiment-rct/geo_split_v3_weekly.csv'

N_SIMULATIONS         = 1000
TEST_WEEKS_LIST       = [8,12,16]              # test durations to evaluate (weeks)
PRE_PERIOD_WEEKS      = 8
EFFECT_SIZES          = [3.0, 5.0, 10.0, 20.0]  # hypothesised lift in %
ALPHA                 = 0.05
POWER_THRESHOLD       = 0.85              # >85% of sims must reject H0

METRIC                = 'bookings'           # metric column to analyse: 'bookings' or 'sales'

N_STRATA              = 4                 # k-means clusters for stratification
STRAT_VARS            = ['mean_bookings', 'std_bookings', 'trend_slope']

BALANCE_THRESHOLD     = 10.0              # max abs % diff in mean bookings
RANDOM_STATE          = 42

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

print("=" * 80)
print("GEO SPLIT — MONTE CARLO RCT POWER ANALYSIS")
print("Central Control Inc. Methodology")
print("=" * 80)

print("\n[LOAD] Reading DMA-level weekly data...")
raw = pd.read_csv(CSV_PATH, parse_dates=['week'])

# Use mapped DMAs only and aggregate postal-code rows up to DMA level
raw = raw[raw['dma_mapped'] == True]
weekly_data = (
    raw.groupby(['week', 'dma_description'], sort=True)[[METRIC]]
    .sum()
    .reset_index()
    .rename(columns={'dma_description': 'dma'})
)

unique_weeks = weekly_data['week'].nunique()
print(f"Unique weeks : {unique_weeks}")
print(f"Unique DMAs  : {weekly_data['dma'].nunique()}")

# ─── DMA COVERAGE FILTER ──────────────────────────────────────────────────────
# Keep only DMAs that appear in every week of the dataset (100% coverage).
# This guarantees no DMA drops out of any simulation window, maximising power.
# If 100% is too strict we step down in 1pp increments, but never drop >20% of DMAs.

_total_dmas = weekly_data['dma'].nunique()
_max_drop   = int(np.floor(_total_dmas * 0.20))   # hard cap: drop at most 20%

_coverage = (
    weekly_data.groupby('dma')['week']
    .nunique()
    .rename('weeks_present')
    .reset_index()
    .assign(coverage=lambda d: d['weeks_present'] / unique_weeks)
    .sort_values('coverage', ascending=False)
    .reset_index(drop=True)
)

# Start at 100% and step down 1pp at a time until within the 20% drop cap
_threshold = 0.0
for _t in np.arange(1.0, 0.0, -0.01):
    _n_fail = (_coverage['coverage'] < _t).sum()
    if _n_fail <= _max_drop:
        _threshold = _t
        break

_keep_dmas  = _coverage.loc[_coverage['coverage'] >= _threshold, 'dma']
_n_dropped  = _total_dmas - len(_keep_dmas)

print(f"\n[DMA FILTER] Coverage threshold : {_threshold:.0%}")
print(f"             DMAs kept          : {len(_keep_dmas)} / {_total_dmas}  ({_n_dropped} dropped, {_n_dropped/_total_dmas:.1%})")
print(f"             Coverage range (kept): {_coverage.loc[_coverage['dma'].isin(_keep_dmas), 'coverage'].min():.1%} – {_coverage['coverage'].max():.1%}")

if _n_dropped > 0:
    _dropped_dmas = _coverage.loc[_coverage['coverage'] < _threshold, ['dma', 'weeks_present', 'coverage']].copy()
    _dropped_dmas = _dropped_dmas.sort_values('weeks_present')
    print(f"\n             Dropped DMAs:")
    print(_dropped_dmas.to_string(index=False))
    import os as _os_excl
    EXCL_PATH = _os_excl.path.join(_os_excl.path.dirname(_os_excl.path.abspath(__file__)), 'dma_excluded.csv')
    _dropped_dmas.to_csv(EXCL_PATH, index=False)
    print(f"\n[SAVED] {EXCL_PATH}")

weekly_data = weekly_data[weekly_data['dma'].isin(_keep_dmas)].copy()
print(f"\n             Rows after filter : {len(weekly_data):,}")
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def stratified_randomization(weekly_data, strat_vars, n_strata=4, seed=42):
    """
    Stratified randomization for Geo RCT using DMA-level aggregates
    """
    # Aggregate weekly data to DMA level for stratification
    dma_metrics = weekly_data.groupby('dma')[strat_vars].agg(['mean', 'sum']).reset_index()

    # Flatten column names (e.g., 'sales_mean')
    dma_metrics.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                           for col in dma_metrics.columns.values]

    # Use aggregated metrics for stratification
    strat_cols = [col for col in dma_metrics.columns if any(var in col for var in strat_vars)]

    # Standardize stratification variables
    scaler = StandardScaler()
    X = scaler.fit_transform(dma_metrics[strat_cols])

    # Create strata using k-means clustering
    kmeans = KMeans(n_clusters=n_strata, random_state=seed)
    dma_metrics['stratum'] = kmeans.fit_predict(X)

    # Create assignment mapping
    np.random.seed(seed)
    dma_assignments = {}

    for stratum in range(n_strata):
        stratum_dmas = dma_metrics[dma_metrics['stratum'] == stratum]['dma'].values
        n_treat = len(stratum_dmas) // 2

        # Randomly select treatment DMAs
        treat_dmas = np.random.choice(stratum_dmas, n_treat, replace=False)

        for dma in stratum_dmas:
            dma_assignments[dma] = 'Treatment' if dma in treat_dmas else 'Control'

        # Handle odd number of DMAs in stratum
        if len(stratum_dmas) % 2 != 0:
            remaining_dmas = [d for d in stratum_dmas if d not in treat_dmas]
            assign_to = np.random.choice(['Treatment', 'Control'])
            dma_assignments[remaining_dmas[0]] = assign_to

    # Apply assignments back to weekly data
    weekly_data['assignment'] = weekly_data['dma'].map(dma_assignments)

    return weekly_data

# Example usage
strat_vars = [METRIC]
assignments = stratified_randomization(weekly_data, strat_vars, n_strata=4, seed=42)

print(assignments[['week', 'dma', METRIC, 'assignment']].head(20))

# ─── FINAL DMA SPLIT ──────────────────────────────────────────────────────────
dma_split = (
    assignments.groupby(['dma', 'assignment'], sort=True)
    .agg(
        total          =(METRIC, 'sum'),
        avg_weekly     =(METRIC, 'mean'),
        weeks_active   =('week', 'nunique'),
    )
    .reset_index()
    .rename(columns={'total': f'total_{METRIC}', 'avg_weekly': f'avg_weekly_{METRIC}'})
    .sort_values(['assignment', 'dma'])
)

# Summary balance check
for grp in ['Treatment', 'Control']:
    sub = dma_split[dma_split['assignment'] == grp]
    print(f"\n{grp}: {len(sub)} DMAs | "
          f"total {METRIC} = {sub[f'total_{METRIC}'].sum():,.0f} | "
          f"avg weekly {METRIC} = {sub[f'avg_weekly_{METRIC}'].sum():,.1f}")

import os as _os
SPLIT_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dma_final_split.csv')
dma_split.to_csv(SPLIT_PATH, index=False)
print(f"\n[SAVED] {SPLIT_PATH}")

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

def run_power_simulation(weekly_data, effect_sizes=[0.05, 0.08, 0.1, 0.2, 0.8],
                         test_weeks=[6, 8, 10, 12], n_sims=1000,
                         alpha=0.05, metric='sales'):
    """
    Run power simulation for geographic RCT using state-level data
    """
    results = []
    for effect in effect_sizes:
        for weeks in test_weeks:
            # Run simulations in parallel
            sim_results = Parallel(n_jobs=-1)(
                delayed(single_simulation)(
                    weekly_data, effect, weeks, alpha, metric
                ) for _ in range(n_sims)
            )
            power = np.mean(sim_results)
            results.append({
                'effect_size': effect,
                'test_weeks': weeks,
                'power': power
            })
    return pd.DataFrame(results)

def single_simulation(weekly_data, effect, weeks, alpha, metric='sales'):
    """
    Single simulation iteration
    """
    # Convert week to datetime for easier handling
    weekly_data_copy = weekly_data.copy()
    weekly_data_copy['week_date'] = pd.to_datetime(weekly_data_copy['week'])
    
    # Get unique weeks sorted
    unique_weeks = sorted(weekly_data_copy['week_date'].unique())
    
    # Sample random test window (ensure space for 8-week pre-period)
    if len(unique_weeks) <= weeks + 8:
        return False  # Not enough data
    
    start_idx = np.random.randint(8, len(unique_weeks) - weeks)
    start_week = unique_weeks[start_idx]
    end_week = unique_weeks[start_idx + weeks - 1]
    
    pre_start = unique_weeks[start_idx - 8]
    # print('Start week:', start_week, 'END week:', end_week, 'PRE week:', pre_start)
    
    # Extract pre and test periods
    pre_data = weekly_data_copy[
        (weekly_data_copy['week_date'] >= pre_start) & 
        (weekly_data_copy['week_date'] < start_week)
    ].copy()
    
    test_data = weekly_data_copy[
        (weekly_data_copy['week_date'] >= start_week) & 
        (weekly_data_copy['week_date'] <= end_week)
    ].copy()
    
    # Random assignment of DMAs to treatment and control
    dmas = weekly_data_copy['dma'].unique()
    treatment_dmas = np.random.choice(dmas, len(dmas) // 2, replace=False)

    # Aggregate weekly metric by DMA
    pre_avg = pre_data.groupby('dma')[metric].mean().reset_index(name='pre_avg')
    test_avg = test_data.groupby('dma')[metric].mean().reset_index(name='test_avg')

    # Assign treatment groups
    treat_set = set(treatment_dmas)
    test_avg['group'] = test_avg['dma'].apply(lambda x: 'T' if x in treat_set else 'C')

    # Merge pre and test periods
    merged = pre_avg.merge(test_avg[['dma', 'test_avg', 'group']], on='dma')

    # Work in log1p space throughout
    merged['log_pre']  = np.log1p(merged['pre_avg'])
    merged['log_test'] = np.log1p(merged['test_avg'])

    # Apply effect additively in log space — gives exactly log(1+effect) signal,
    # avoiding the compression that occurs when multiplying raw counts then logging
    merged.loc[merged['group'] == 'T', 'log_test'] += np.log1p(effect)

    merged['diff'] = merged['log_test'] - merged['log_pre']

    # Run t-test on log-diff between groups
    t_vals = merged[merged['group'] == 'T']['diff']
    c_vals = merged[merged['group'] == 'C']['diff']

    # Need ≥2 per group for Welch's t-test to estimate variance
    if len(t_vals) > 1 and len(c_vals) > 1:
        t_stat, p_val = ttest_ind(t_vals, c_vals, equal_var=False)
        return p_val < alpha
    return False

# ─── DIAGNOSTIC ──────────────────────────────────────────────────────────────
print("\n--- SINGLE-SIM DIAGNOSTIC ---")
_wd = weekly_data.copy()
_wd['week_date'] = pd.to_datetime(_wd['week'])
_unique = sorted(_wd['week_date'].unique())
_effect = EFFECT_SIZES[0] / 100
_weeks  = TEST_WEEKS_LIST[0]
print(f"Unique weeks : {len(_unique)}  (need > {_weeks + 8})")
print(f"Unique DMAs  : {_wd['dma'].nunique()}")
print(f"Metric column: '{METRIC}'  (columns in data: {list(_wd.columns)})")
if len(_unique) <= _weeks + 8:
    print("FAIL — early return: not enough weeks in data")
else:
    np.random.seed(99)
    _si = np.random.randint(8, len(_unique) - _weeks)
    _pre = _wd[(_wd['week_date'] >= _unique[_si - 8]) & (_wd['week_date'] < _unique[_si])]
    _tst = _wd[(_wd['week_date'] >= _unique[_si]) & (_wd['week_date'] <= _unique[_si + _weeks - 1])]
    _dmas  = _wd['dma'].unique()
    _treat = np.random.choice(_dmas, len(_dmas) // 2, replace=False)
    _pre_avg = _pre.groupby('dma')[METRIC].mean().reset_index(name='pre_avg')
    _tst_avg = _tst.groupby('dma')[METRIC].mean().reset_index(name='test_avg')
    _treat_set = set(_treat)
    _tst_avg['group'] = _tst_avg['dma'].apply(lambda x: 'T' if x in _treat_set else 'C')
    _merged = _pre_avg.merge(_tst_avg[['dma', 'test_avg', 'group']], on='dma')
    _merged['log_pre']  = np.log1p(_merged['pre_avg'])
    _merged['log_test'] = np.log1p(_merged['test_avg'])
    _merged.loc[_merged['group'] == 'T', 'log_test'] += np.log1p(_effect)
    _merged['diff'] = _merged['log_test'] - _merged['log_pre']
    _tv = _merged[_merged['group'] == 'T']['diff']
    _cv = _merged[_merged['group'] == 'C']['diff']
    print(f"T DMAs: {len(_tv)}, C DMAs: {len(_cv)}")
    print(f"Mean log-diff  T={_tv.mean():.4f}  C={_cv.mean():.4f}")
    if len(_tv) > 1 and len(_cv) > 1:
        from scipy.stats import ttest_ind as _tt
        _, _p = _tt(_tv, _cv, equal_var=False)
        print(f"p-value: {_p:.4f}  ({'REJECT H0' if _p < ALPHA else 'fail to reject'})")
    else:
        print("FAIL — fewer than 2 DMAs in one group; t-test impossible")
print("--- END DIAGNOSTIC ---\n")

# ─── RUN ─────────────────────────────────────────────────────────────────────
power_results = run_power_simulation(
    weekly_data,
    effect_sizes=[e / 100 for e in EFFECT_SIZES],  # convert % to decimal
    test_weeks=TEST_WEEKS_LIST,
    n_sims=N_SIMULATIONS,
    alpha=ALPHA,
    metric=METRIC,
)
print(power_results)

import os as _os2
POWER_PATH = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), 'power_results.csv')
power_results.to_csv(POWER_PATH, index=False)
print(f"\n[SAVED] {POWER_PATH}")

# ─── MEAN DISTRIBUTION PLOT (fixed DMA split) ─────────────────────────────────
# For the final T/C assignment, sample N random test windows and record the
# per-window mean log-diff for each group.  Two panels:
#   Left  — effect = 0  (null): distributions should overlap  → shows balance
#   Right — effect = smallest chosen size: shows signal vs noise

import matplotlib
import matplotlib.pyplot as plt

def _sim_means_fixed_split(weekly_data_assigned, weeks, effect, metric, n_sims=1000, seed=0):
    """Collect per-window T/C mean log-diffs using the FIXED DMA assignment."""
    wd = weekly_data_assigned.copy()
    wd['week_date'] = pd.to_datetime(wd['week'])
    unique_wks = sorted(wd['week_date'].unique())
    assign_map = (
        wd[['dma', 'assignment']].drop_duplicates()
        .assign(group=lambda d: d['assignment'].map({'Treatment': 'T', 'Control': 'C'}))
        [['dma', 'group']]
    )
    np.random.seed(seed)
    t_means, c_means = [], []
    for _ in range(n_sims):
        if len(unique_wks) <= weeks + 8:
            break
        si = np.random.randint(8, len(unique_wks) - weeks)
        pre = wd[(wd['week_date'] >= unique_wks[si - 8]) & (wd['week_date'] < unique_wks[si])]
        tst = wd[(wd['week_date'] >= unique_wks[si]) & (wd['week_date'] <= unique_wks[si + weeks - 1])]
        pre_avg = pre.groupby('dma')[metric].mean().reset_index(name='pre_avg')
        tst_avg = tst.groupby('dma')[metric].mean().reset_index(name='test_avg')
        merged = pre_avg.merge(tst_avg, on='dma').merge(assign_map, on='dma')
        merged['log_pre']  = np.log1p(merged['pre_avg'])
        merged['log_test'] = np.log1p(merged['test_avg'])
        merged.loc[merged['group'] == 'T', 'log_test'] += np.log1p(effect)
        merged['diff'] = merged['log_test'] - merged['log_pre']
        tv = merged[merged['group'] == 'T']['diff']
        cv = merged[merged['group'] == 'C']['diff']
        if len(tv) > 1 and len(cv) > 1:
            t_means.append(tv.mean())
            c_means.append(cv.mean())
    return np.array(t_means), np.array(c_means)

_plot_weeks  = TEST_WEEKS_LIST[0]
_plot_effect = EFFECT_SIZES[0] / 100
_n_plot_sims = 500

print(f"\n[PLOT] Simulating mean distributions ({_n_plot_sims} windows, "
      f"fixed split, {_plot_weeks}w, effect={EFFECT_SIZES[0]}%)...")

_t0, _c0 = _sim_means_fixed_split(assignments, _plot_weeks, 0.0,          METRIC, _n_plot_sims, seed=1)
_t1, _c1 = _sim_means_fixed_split(assignments, _plot_weeks, _plot_effect, METRIC, _n_plot_sims, seed=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (tm, cm, title) in zip(axes, [
    (_t0, _c0, f'Null (effect = 0%)  |  {_plot_weeks}-week windows'),
    (_t1, _c1, f'Effect = {EFFECT_SIZES[0]}%  |  {_plot_weeks}-week windows'),
]):
    _all = np.concatenate([tm, cm])
    bins = np.linspace(_all.min(), _all.max(), 40)
    ax.hist(cm, bins=bins, alpha=0.55, color='steelblue', label=f'Control  (μ={cm.mean():.3f})', density=True)
    ax.hist(tm, bins=bins, alpha=0.55, color='tomato',    label=f'Treatment (μ={tm.mean():.3f})', density=True)
    ax.axvline(cm.mean(), color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(tm.mean(), color='tomato',    linestyle='--', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Mean log-diff  (log(1+test) − log(1+pre))')
    ax.set_ylabel('Density')
    ax.legend()

fig.suptitle(
    f'Distribution of Group Means Across Simulation Windows\n'
    f'Fixed DMA Split  |  Metric: {METRIC}  |  {_n_plot_sims} windows',
    fontsize=12
)
fig.tight_layout()

import os as _os3
PLOT_PATH = _os3.path.join(_os3.path.dirname(_os3.path.abspath(__file__)), 'sim_mean_distributions.png')
fig.savefig(PLOT_PATH, dpi=150)
print(f"[SAVED] {PLOT_PATH}")
# ─────────────────────────────────────────────────────────────────────────────

# ─── FIXED-SPLIT POWER ANALYSIS ───────────────────────────────────────────────
# Re-runs the power simulation but using the FIXED stratified assignment
# (from assignments) instead of re-randomising each iteration.
# This gives the actual power of your chosen split, not the average over all
# possible splits.

print("\n" + "=" * 80)
print("FIXED-SPLIT POWER ANALYSIS")
print("=" * 80)

from scipy.stats import ttest_ind as _ttest_fixed

def _single_sim_fixed_split(weekly_data_assigned, effect, weeks, alpha, metric, seed=None):
    """One simulation using the FIXED T/C assignment from stratified_randomization."""
    wd = weekly_data_assigned.copy()
    wd['week_date'] = pd.to_datetime(wd['week'])
    unique_wks = sorted(wd['week_date'].unique())

    if len(unique_wks) <= weeks + 8:
        return False

    rng = np.random.default_rng(seed)
    si  = rng.integers(8, len(unique_wks) - weeks)

    pre = wd[(wd['week_date'] >= unique_wks[si - 8]) & (wd['week_date'] < unique_wks[si])]
    tst = wd[(wd['week_date'] >= unique_wks[si])     & (wd['week_date'] <= unique_wks[si + weeks - 1])]

    pre_avg = pre.groupby('dma')[metric].mean().reset_index(name='pre_avg')
    tst_avg = tst.groupby('dma')[metric].mean().reset_index(name='test_avg')

    # Use fixed assignment — map Treatment→T, Control→C
    assign = (
        wd[['dma', 'assignment']].drop_duplicates()
        .assign(group=lambda d: d['assignment'].map({'Treatment': 'T', 'Control': 'C'}))
        [['dma', 'group']]
    )

    merged = pre_avg.merge(tst_avg, on='dma').merge(assign, on='dma')
    if merged.empty:
        return False

    merged['log_pre']  = np.log1p(merged['pre_avg'])
    merged['log_test'] = np.log1p(merged['test_avg'])
    merged.loc[merged['group'] == 'T', 'log_test'] += np.log1p(effect)
    merged['diff'] = merged['log_test'] - merged['log_pre']

    t_vals = merged[merged['group'] == 'T']['diff']
    c_vals = merged[merged['group'] == 'C']['diff']
    if len(t_vals) < 2 or len(c_vals) < 2:
        return False

    _, p = _ttest_fixed(t_vals, c_vals, equal_var=False)
    return p < alpha


from joblib import Parallel as _ParallelFixed, delayed as _delayedFixed

_fixed_rows = []
for _eff_pct in EFFECT_SIZES:
    _eff = _eff_pct / 100
    for _tw in TEST_WEEKS_LIST:
        _sigs = _ParallelFixed(n_jobs=-1)(
            _delayedFixed(_single_sim_fixed_split)(
                assignments, _eff, _tw, ALPHA, METRIC, seed=i
            )
            for i in range(N_SIMULATIONS)
        )
        _pwr = float(np.mean(_sigs))
        _fixed_rows.append({
            'effect_pct': _eff_pct,
            'test_weeks': _tw,
            'power_fixed_split': _pwr,
        })
        print(f"  effect={_eff_pct:>5.1f}%  weeks={_tw:>2}  power={_pwr:.3f}"
              f"  {'✓' if _pwr >= POWER_THRESHOLD else '✗'}")

fixed_power_df = pd.DataFrame(_fixed_rows)

# Merge with random-split results for easy comparison
# power_results uses effect_size (decimal); fixed_power_df uses effect_pct (%)
_fixed_merge = fixed_power_df.assign(effect_size=fixed_power_df['effect_pct'] / 100)
_compare = power_results.rename(columns={'power': 'power_random_split'}).merge(
    _fixed_merge[['effect_size', 'test_weeks', 'power_fixed_split']],
    on=['effect_size', 'test_weeks'], how='left'
)
print("\nComparison — random split vs fixed split:")
print(_compare.to_string(index=False))

import os as _os4
FIXED_POWER_PATH = _os4.path.join(_os4.path.dirname(_os4.path.abspath(__file__)), 'power_results_fixed_split.csv')
fixed_power_df.to_csv(FIXED_POWER_PATH, index=False)
print(f"\n[SAVED] {FIXED_POWER_PATH}")

# ── FIXED-SPLIT POWER PLOT ────────────────────────────────────────────────────
_fp_pivot = fixed_power_df.pivot(index='test_weeks', columns='effect_pct', values='power_fixed_split')

_fig_fp, _axes_fp = plt.subplots(1, 2, figsize=(14, 5))

# Left: heatmap
_ax = _axes_fp[0]
_im = _ax.imshow(_fp_pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
_ax.set_xticks(range(len(_fp_pivot.columns)))
_ax.set_xticklabels([f'{c:.0f}%' for c in _fp_pivot.columns])
_ax.set_yticks(range(len(_fp_pivot.index)))
_ax.set_yticklabels([f'{r}w' for r in _fp_pivot.index])
_ax.set_xlabel('Effect size (lift %)')
_ax.set_ylabel('Test duration (weeks)')
_ax.set_title('Power heatmap\n(fixed DMA split — t-test)')
for _i in range(len(_fp_pivot.index)):
    for _j in range(len(_fp_pivot.columns)):
        _val = _fp_pivot.values[_i, _j]
        _ax.text(_j, _i, f'{_val:.0%}', ha='center', va='center', fontsize=9,
                 color='black' if 0.2 < _val < 0.8 else 'white')
plt.colorbar(_im, ax=_ax, label='Power')

# Right: power curves per test duration
_ax = _axes_fp[1]
for _tw in sorted(fixed_power_df['test_weeks'].unique()):
    _row = fixed_power_df[fixed_power_df['test_weeks'] == _tw].sort_values('effect_pct')
    _ax.plot(_row['effect_pct'], _row['power_fixed_split'], marker='o', label=f'{_tw}w')
_ax.axhline(POWER_THRESHOLD, color='black', linestyle='--', linewidth=1,
            label=f'{POWER_THRESHOLD:.0%} threshold')
_ax.set_xlabel('Effect size (lift %)')
_ax.set_ylabel('Power')
_ax.set_title('Power curves by test duration\n(fixed DMA split — t-test)')
_ax.legend()
_ax.set_ylim(0, 1.05)

_fig_fp.suptitle(
    f'Fixed-Split Power Profile  |  Metric: {METRIC}  |  {N_SIMULATIONS} simulations',
    fontsize=12
)
_fig_fp.tight_layout()

FIXED_PLOT_PATH = _os4.path.join(_os4.path.dirname(_os4.path.abspath(__file__)), 'power_fixed_split_plot.png')
_fig_fp.savefig(FIXED_PLOT_PATH, dpi=150)
plt.close(_fig_fp)
print(f"[SAVED] {FIXED_PLOT_PATH}")
# ─────────────────────────────────────────────────────────────────────────────

# ─── DIAGNOSTIC ──────────────────────────────────────────────────────────────
