library(tidyverse)
library(lubridate)
library(CausalImpact)
library(patchwork)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR   <- dirname(rstudioapi::getSourceEditorContext()$path)
pre_start  <- as.Date("2025-02-01")
post_start <- as.Date("2026-01-07")   # 8 weeks ago from 2026-03-04
post_end   <- as.Date("2026-03-04")
N_SIMS     <- 200

# ── LOAD & PREP ────────────────────────────────────────────────────────────────
weekly_raw <- read_csv(file.path(BASE_DIR, "geo_split_v3_weekly.csv"),
                       show_col_types = FALSE) |>
  filter(dma_mapped == TRUE) |>
  mutate(week = as.Date(week))

split_df <- read_csv(file.path(BASE_DIR, "dma_final_split.csv"),
                     show_col_types = FALSE) |>
  select(dma, assignment) |>
  distinct()

# Aggregate postal codes → DMA, attach T/C assignment
dma_weekly <- weekly_raw |>
  group_by(week, dma_description) |>
  summarise(bookings = sum(bookings), .groups = "drop") |>
  rename(dma = dma_description) |>
  inner_join(split_df, by = "dma")

# Weekly totals per group → wide format (Treatment | Control)
df <- dma_weekly |>
  filter(week >= pre_start & week <= post_end) |>
  group_by(week, assignment) |>
  summarise(bookings = sum(bookings), .groups = "drop") |>
  pivot_wider(names_from = assignment, values_from = bookings) |>
  arrange(week) |>
  select(week, Treatment, Control)

cat("Weeks in series:", nrow(df), "\n")
print(df)

# ── NORMALISE TO PRE-PERIOD MEAN (index = 100) ────────────────────────────────
# Removes level/trend offset between T and C so CausalImpact compares
# relative movements only. Without this, a persistently smaller Treatment
# group biases all rel_effect estimates negative.
pre_period  <- c(1, sum(df$week < post_start))
post_period <- c(pre_period[2] + 1, nrow(df))

pre_mean_T <- mean(df$Treatment[pre_period[1]:pre_period[2]], na.rm = TRUE)
pre_mean_C <- mean(df$Control[pre_period[1]:pre_period[2]],   na.rm = TRUE)

df_norm <- df |>
  mutate(
    Treatment = Treatment / pre_mean_T * 100,
    Control   = Control   / pre_mean_C * 100
  )

cat("Pre-period mean — Treatment:", round(pre_mean_T), " Control:", round(pre_mean_C), "\n")

# ── SINGLE CAUSAL IMPACT (actual placebo window) ───────────────────────────────
impact <- CausalImpact(select(df_norm, Treatment, Control), pre_period, post_period)

summary(impact)
summary(impact, "report")
plot(impact)

# ── SIMULATION ────────────────────────────────────────────────────────────────
# For each iteration: pick a random event row, run CausalImpact, record results.
# Post = 8 weeks, pre = at least 26 weeks. No real treatment → FPR should be ~5%.

run_one <- function(df, event_row) {
  pp  <- c(1, event_row - 1)
  po  <- c(event_row, event_row + 7)

  # Normalise each series to its own pre-period mean
  mu_T <- mean(df$Treatment[pp[1]:pp[2]], na.rm = TRUE)
  mu_C <- mean(df$Control[pp[1]:pp[2]],   na.rm = TRUE)
  df_n <- df |> mutate(Treatment = Treatment / mu_T * 100,
                       Control   = Control   / mu_C * 100)

  imp <- tryCatch(CausalImpact(select(df_n, Treatment, Control), pp, po),
                  error = function(e) NULL)
  if (is.null(imp)) return(NULL)

  rmse <- sqrt(mean((imp$series$response - imp$series$point.pred)^2, na.rm = TRUE))

  list(
    rmse       = rmse,
    rel_effect = imp$summary["Average", "RelEffect"],
    p_value    = imp$summary["Average", "p"]
  )
}

set.seed(42)
valid_rows  <- 27:(nrow(df) - 8)          # pre ≥ 26w, post = 8w
event_rows  <- sample(valid_rows, N_SIMS, replace = TRUE)
raw_results <- lapply(event_rows, \(er) run_one(df, er))
results     <- bind_rows(Filter(Negate(is.null), raw_results))

# ── RESULTS ───────────────────────────────────────────────────────────────────
cat("\nFalse-positive rate (p < 0.05):", mean(results$p_value < 0.05), "\n")
cat("Median RMSE:", median(results$rmse), "\n")
cat("\nRel effect percentiles (%):\n")
print(quantile(results$rel_effect * 100, probs = c(0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)))

# ── PLOTS ─────────────────────────────────────────────────────────────────────
p1 <- ggplot(results, aes(p_value)) +
  geom_histogram(bins = 30, fill = "steelblue") +
  geom_vline(xintercept = 0.05, linetype = "dashed", colour = "red") +
  labs(title = "p-value distribution", x = "p-value", y = "count") +
  theme_minimal()

p2 <- ggplot(results, aes(rel_effect * 100)) +
  geom_histogram(bins = 30, fill = "tomato") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Relative effect (%)", x = "rel effect %", y = "count") +
  theme_minimal()

p3 <- ggplot(results, aes(rmse)) +
  geom_histogram(bins = 30, fill = "seagreen") +
  labs(title = "Pre-period RMSE", x = "RMSE", y = "count") +
  theme_minimal()

ggsave(file.path(BASE_DIR, "placebo_sim_results.png"),
       p1 + p2 + p3, width = 14, height = 4, dpi = 150)
