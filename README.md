# Geo Experimentation — Split Design & Analysis

This repo contains two Jupyter notebooks for designing geo-based randomised controlled trials (RCTs). Each notebook guides a non-technical user through creating a balanced Treatment / Control split, validating it with a placebo test, and estimating statistical power before the experiment launches.

```
notebooks/   Jupyter notebooks (US DMA split, UK postcode split)
scripts/     Python and R analysis scripts
data/        Input and output CSV files
plots/       Generated charts and simulation outputs
```

---

## Notebooks

### `notebooks/Geo-split-notebook.ipynb` — US DMA Split

Splits **US media markets (DMAs)** into Treatment and Control groups using stratified randomisation.

| Step | What it does |
|------|--------------|
| 0–2  | Load Braintree transaction data, filter to US, remove refunds |
| 3–4  | Roll up to weekly bookings per ZIP code, then map ZIPs → DMAs |
| 5–6  | Aggregate to weekly DMA level; drop DMAs with sparse history |
| 7    | Stratified k-means split: similar-sized DMAs are randomised within clusters |
| 8    | Balance check — compare pre-period booking volumes across groups |
| 9    | Placebo simulation — slide an 8-week window across history to measure natural T/C variance |
| 10   | Save `data/dma_final_split.csv` for campaign targeting |
| 11   | Power heatmap — probability of detecting a true lift at different effect sizes and durations |

**Key outputs:** `data/dma_final_split.csv`, `data/power_results_fixed_split.csv`, `plots/power_fixed_split_plot.png`

---

### `notebooks/GEO_split_UK_postcodes.ipynb` — UK Postcode Area Split

Splits **UK postcode areas** (`NW`, `SW`, `E`, etc.) into Treatment and Control groups.

UK postcodes are parsed from full format (e.g. `NW3 4ED`) down to the area level (`NW`), which gives naturally-sized geographic units suitable for geo testing.

| Step | What it does |
|------|--------------|
| 0    | Config cell — set `N_CLUSTERS`, `PRE_PERIOD_WEEKS`, and `EXCLUDE_AREAS` here |
| 1–2  | Load Braintree data, filter to UK, extract district and area from postcode |
| 3–4  | Roll up to weekly bookings per area; bar chart of area volumes |
| 5–6  | K-means clustering on booking size/variance; random T/C assignment within clusters |
| 7    | Normalise each area to its own 8-week pre-period baseline (index = 100) |
| 8–9  | Trend charts — Treatment vs Control at group level and by cluster |
| 10   | Save `data/uk_area_split.csv` |
| 11   | Placebo simulation — sliding 8-week window distribution of T−C index difference |
| 12   | Power matrix — detection probability across effect sizes (5–30%) and durations (4–12 weeks) |

**Key outputs:** `data/uk_area_split.csv`, `plots/uk_placebo_sim.png`, `plots/uk_power_matrix.png`

---

## How to run

1. Export transactions from Braintree and place the CSV in the **repo root** (it is gitignored)
2. Open a notebook from the `notebooks/` folder in VS Code
3. Select a Python environment with `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`
4. Run **Step 0** first, then execute cells top to bottom
5. Only the **Config cell** needs editing to adjust split parameters

> The Braintree CSV is excluded from this repo via `.gitignore` — never commit raw transaction data.

---

## R placebo check (`scripts/r_split_checks.r`)

For the US split, an additional R script runs a formal **CausalImpact** placebo test on the final DMA assignment. It simulates 200 random event dates, runs Bayesian structural time-series on each, and reports the false-positive rate and distribution of relative effects.

Requires R packages: `tidyverse`, `lubridate`, `CausalImpact`, `patchwork`
