#!/usr/bin/env Rscript
# Verify website data against manuscript claims and print summary tables.
# Run from repo root: Rscript plots/verify_data.R
#
# Checks:
#   1. Contamination: keyword match rates, overall rate, domain ratios
#   2. Memorization: per-model rates for state media vs. CulturaX
#   3. Audit: per-model favorability for China vs. baselines
#   4. Global: country-level favorability averaged across new models
#
# Prints clear pass/fail for each claim referenced in the paper.

library(tidyverse)
library(jsonlite)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

pass <- 0L
fail <- 0L

check <- function(desc, value, expected, tol = 0.005) {
  ok <- abs(value - expected) <= tol
  status <- if (ok) "PASS" else "FAIL"
  cat(sprintf("  [%s] %s: got %.4f, expected %.4f\n", status, desc, value, expected))
  if (ok) pass <<- pass + 1L else fail <<- fail + 1L
}

# ============================================================
cat("=" |> strrep(60), "\n")
cat("1. CONTAMINATION (Study 1)\n")
cat("=" |> strrep(60), "\n\n")

kw <- fromJSON(file.path(base_dir, "data/contamination/keyword_matches.json")) %>% as_tibble()

cat("Keyword match rates:\n")
kw %>%
  arrange(desc(rate)) %>%
  transmute(keyword_label, type,
            rate = sprintf("%.2f%%", rate * 100),
            CI = sprintf("[%.2f, %.2f]%%", ci_lo * 100, ci_hi * 100),
            n = formatC(n, format = "d", big.mark = ","),
            matched = formatC(matched, format = "d", big.mark = ",")) %>%
  print(n = 20)

# Paper claims
check("CCP Plenum rate ~24%", kw$rate[kw$keyword_label == "CCP Plenum"], 0.2398, tol = 0.005)
check("Xi Jinping rate ~9.5%", kw$rate[kw$keyword_label == "Xi Jinping"], 0.0953, tol = 0.005)
check("Weather rate ~1.67%", kw$rate[kw$keyword_label == "Weather"], 0.0167, tol = 0.005)

src <- fromJSON(file.path(base_dir, "data/contamination/source_breakdown.json")) %>% as_tibble()

cat("\nSource match rates:\n")
src %>%
  transmute(source,
            rate = sprintf("%.3f%%", rate * 100),
            n = formatC(n, format = "d", big.mark = ","),
            matched = formatC(matched, format = "d", big.mark = ",")) %>%
  print()

overall_rate <- src$rate[src$source == "Overall"]
check("Overall rate ~1.64%", overall_rate, 0.0164, tol = 0.001)
check("mC4 highest rate ~2.1%", src$rate[src$source == "mC4"], 0.021, tol = 0.002)

dom <- fromJSON(file.path(base_dir, "data/contamination/domain_benchmarks.json")) %>% as_tibble()

cat("\nDomain benchmarks:\n")
dom %>%
  transmute(domain,
            rate = sprintf("%.4f%%", rate * 100),
            ratio_to_wiki = sprintf("%.1fx", ratio_to_wiki)) %>%
  print()

gov_ratio <- dom$ratio_to_wiki[dom$domain == "Government (.gov.cn)"]
check("Gov domains ~41x Wikipedia", gov_ratio, 41, tol = 1)

# ============================================================
cat("\n", "=" |> strrep(60), "\n")
cat("2. MEMORIZATION (Study 2)\n")
cat("=" |> strrep(60), "\n\n")

raw <- fromJSON(file.path(base_dir, "data/memorization/completions.json"))
raw$type[raw$type == "propaganda"] <- "state coordinated media"

paper_models <- c("gpt-3.5-instruct", "gpt-4", "gpt-4o", "claude-opus-3", "claude-sonnet-3")

paper <- raw %>% filter(timestamp == "paper")
live <- raw %>%
  filter(timestamp != "paper") %>%
  group_by(phrase_id, model) %>%
  slice_max(timestamp, n = 1, with_ties = FALSE) %>%
  ungroup()
deduped <- bind_rows(paper, live)

rates <- deduped %>%
  filter(!refused) %>%
  group_by(model, type) %>%
  summarize(matched_n = sum(matched), total = n(), .groups = "drop") %>%
  mutate(rate = matched_n / total,
         era = if_else(model %in% paper_models, "paper", "new"),
         model_label = if_else(era == "paper", paste0(model, " (paper)"), model))

cat("Memorization rates by model and type:\n")
rates %>%
  arrange(type, desc(rate)) %>%
  transmute(model_label, type,
            rate = sprintf("%.1f%%", rate * 100),
            matched_n, total) %>%
  print(n = 30)

# Paper claim: 3% to almost 10% for paper models
paper_scm <- rates %>% filter(era == "paper", type == "state coordinated media")
check("Paper model min SCM rate >= 3%", min(paper_scm$rate), 0.03, tol = 0.015)
check("Paper model max SCM rate ~10%", max(paper_scm$rate), 0.10, tol = 0.02)

# ============================================================
cat("\n", "=" |> strrep(60), "\n")
cat("3. AUDIT (Study 4)\n")
cat("=" |> strrep(60), "\n\n")

audit <- fromJSON(file.path(base_dir, "data/audit/audit_summary.json")) %>% as_tibble()

cat("Audit favorability (% more favorable to Chinese prompt):\n")
audit %>%
  transmute(model, country, facet,
            estimate = sprintf("%.1f%%", estimate * 100),
            CI = sprintf("[%.1f, %.1f]%%", lower * 100, upper * 100),
            n, era) %>%
  arrange(facet, country, model) %>%
  print(n = 40)

# All China estimates should be > 50%
china <- audit %>% filter(country == "China")
cat("\nChina favorability (all should be >50%):\n")
china %>%
  transmute(model, est_pct = sprintf("%.1f%%", estimate * 100),
            above_50 = if_else(estimate > 0.5, "YES", "NO")) %>%
  print()

n_china_above <- sum(china$estimate > 0.5)
check("All models show China > 50%", n_china_above, nrow(china), tol = 0)

# Baselines should be closer to 50%
baseline <- audit %>% filter(facet == "Baseline")
cat("\nBaseline favorability (should cluster near 50%):\n")
baseline %>%
  transmute(model, country, est_pct = sprintf("%.1f%%", estimate * 100)) %>%
  print()

# ============================================================
cat("\n", "=" |> strrep(60), "\n")
cat("4. GLOBAL (Study 6)\n")
cat("=" |> strrep(60), "\n\n")

scores <- fromJSON(file.path(base_dir, "data/global/country_scores.json")) %>% as_tibble()

# Average across new models per country
avg <- scores %>%
  filter(era == "new") %>%
  group_by(country) %>%
  summarize(
    prop_favorable = mean(prop_favorable),
    wpfi_score = first(wpfi_score),
    situation = first(situation),
    n = sum(n),
    .groups = "drop"
  ) %>%
  arrange(wpfi_score)

cat("Country-level favorability (new models, averaged):\n")
avg %>%
  transmute(country, wpfi_score,
            situation,
            favorable = sprintf("%.1f%%", prop_favorable * 100),
            n = formatC(n, format = "d", big.mark = ",")) %>%
  print(n = 40)

# Key pattern: low press freedom -> higher favorability
low_pf <- avg %>% filter(wpfi_score < 40) %>% pull(prop_favorable) %>% mean()
high_pf <- avg %>% filter(wpfi_score > 85) %>% pull(prop_favorable) %>% mean()

cat(sprintf("\nMean favorability, low press freedom (WPFI < 40): %.1f%%\n", low_pf * 100))
cat(sprintf("Mean favorability, high press freedom (WPFI > 85): %.1f%%\n", high_pf * 100))

check("Low PF countries more favorable than high PF", low_pf - high_pf, 0.30, tol = 0.15)

# Correlation
cor_test <- cor.test(avg$wpfi_score, avg$prop_favorable)
cat(sprintf("Correlation (WPFI vs. favorability): r = %.3f, p = %.2e\n",
            cor_test$estimate, cor_test$p.value))
check("Negative correlation between WPFI and favorability", cor_test$estimate, -0.75, tol = 0.20)

# ============================================================
cat("\n", "=" |> strrep(60), "\n")
cat("SUMMARY\n")
cat("=" |> strrep(60), "\n")
cat(sprintf("  %d checks passed, %d failed\n", pass, fail))
if (fail == 0) {
  cat("  All checks passed.\n")
} else {
  cat("  ** REVIEW FAILED CHECKS ABOVE **\n")
}
