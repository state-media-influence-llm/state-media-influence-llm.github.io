#!/usr/bin/env Rscript
# Process Study 1 CulturaX contamination data into JSON for the website
# Input: Three RDS files from study1_culturax/data/
# Output: Four JSON files in data/contamination/

library(tidyverse)
library(jsonlite)

data_dir <- "/Users/ns/workspace/propaganda_llm_gh/code_public/study1_culturax/data"
out_dir <- "/Users/ns/workspace/llm_propaganda_web/data/contamination"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load data ──
key_data <- readRDS(file.path(data_dir, "matched_merged_key_0_319_update.rds"))
overall_data <- readRDS(file.path(data_dir, "matched_merged_overall_0_319_update.rds"))
domain_data <- readRDS(file.path(data_dir, "matched_merged_domains_0_319_update.rds"))

cat("Key data:", nrow(key_data), "rows x", ncol(key_data), "cols\n")
cat("Overall data:", nrow(overall_data), "rows x", ncol(overall_data), "cols\n")
cat("Domain data:", nrow(domain_data), "rows x", ncol(domain_data), "cols\n")

# ── Keyword labels and types ──
keyword_labels <- tribble(
  ~Keyword, ~keyword_label, ~keyword_zh, ~type,
  "xjp",      "Xi Jinping",        "习近平",                "Leaders",
  "mzd",      "Mao Zedong",        "毛泽东",                "Leaders",
  "dxp",      "Deng Xiaoping",     "邓小平",                "Leaders",
  "party",    "Communist Party",   "中国共产党",             "Institutions",
  "npc",      "Natl People's Congress", "人民代表大会",      "Institutions",
  "plenum",   "CCP Plenum",        "中央委员会全体会议",     "Institutions",
  "economy",  "Economy/Development","经济发展",              "Institutions",
  "foreign",  "Foreign Ministry",  "外交部发言人",           "Institutions",
  "weather",  "Weather",           "天气",                  "Not Political",
  "soccer",   "Soccer Scores",     "足球",                  "Not Political"
)

# ── 1. Keyword match rates ──
# Combine party congress keywords into "pc" if present
# Aggregate across all sources, compute Wilson CIs
keyword_rates <- key_data %>%
  group_by(Keyword) %>%
  summarize(
    n = sum(n, na.rm = TRUE),
    matched = sum(count_2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Combine party congress keywords
  bind_rows(
    key_data %>%
      filter(Keyword %in% c("pc_18", "pc_19", "pc_20")) %>%
      summarize(
        Keyword = "pc",
        n = sum(n, na.rm = TRUE),
        matched = sum(count_2, na.rm = TRUE)
      )
  ) %>%
  # Exclude individual pc_XX rows (keep combined "pc")
  filter(!Keyword %in% c("pc_18", "pc_19", "pc_20")) %>%
  mutate(
    rate = matched / n,
    # Wilson score 95% CI
    z = 1.96,
    denom = 1 + z^2 / n,
    center = (rate + z^2 / (2 * n)) / denom,
    margin = z * sqrt((rate * (1 - rate) + z^2 / (4 * n)) / n) / denom,
    ci_lo = pmax(0, center - margin),
    ci_hi = pmin(1, center + margin)
  ) %>%
  select(-z, -denom, -center, -margin)

# Add pc label
keyword_labels_full <- keyword_labels %>%
  bind_rows(tribble(
    ~Keyword, ~keyword_label, ~keyword_zh, ~type,
    "pc", "Party Congress", "全国代表大会", "Institutions"
  ))

keyword_rates <- keyword_rates %>%
  left_join(keyword_labels_full, by = "Keyword") %>%
  filter(!is.na(keyword_label)) %>%
  arrange(type, desc(rate)) %>%
  select(keyword = Keyword, keyword_label, keyword_zh, type, n, matched, rate, ci_lo, ci_hi)

cat("\nKeyword match rates:\n")
keyword_rates %>%
  mutate(rate_pct = sprintf("%.2f%%", rate * 100)) %>%
  select(keyword_label, type, rate_pct, n, matched) %>%
  print(n = 20)

write_json(keyword_rates, file.path(out_dir, "keyword_matches.json"), pretty = TRUE)

# ── 2. Source breakdown ──
source_rates <- overall_data %>%
  group_by(source) %>%
  summarize(
    n = sum(n, na.rm = TRUE),
    matched = sum(count_2, na.rm = TRUE),
    matched_1 = sum(count_1, na.rm = TRUE),
    matched_55 = sum(count_55, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    rate = matched / n,
    rate_1 = matched_1 / n,
    rate_55 = matched_55 / n,
    z = 1.96,
    denom = 1 + z^2 / n,
    center = (rate + z^2 / (2 * n)) / denom,
    margin = z * sqrt((rate * (1 - rate) + z^2 / (4 * n)) / n) / denom,
    ci_lo = pmax(0, center - margin),
    ci_hi = pmin(1, center + margin)
  ) %>%
  select(source, n, matched, rate, ci_lo, ci_hi, rate_1, rate_55)

# Add overall row
overall_agg <- overall_data %>%
  summarize(
    source = "Overall",
    n = sum(n, na.rm = TRUE),
    matched = sum(count_2, na.rm = TRUE)
  ) %>%
  mutate(
    rate = matched / n,
    z = 1.96,
    denom = 1 + z^2 / n,
    center = (rate + z^2 / (2 * n)) / denom,
    margin = z * sqrt((rate * (1 - rate) + z^2 / (4 * n)) / n) / denom,
    ci_lo = pmax(0, center - margin),
    ci_hi = pmin(1, center + margin),
    rate_1 = NA_real_,
    rate_55 = NA_real_
  ) %>%
  select(source, n, matched, rate, ci_lo, ci_hi, rate_1, rate_55)

source_rates <- bind_rows(source_rates, overall_agg) %>%
  arrange(desc(rate))

cat("\nSource breakdown:\n")
source_rates %>%
  mutate(rate_pct = sprintf("%.3f%%", rate * 100)) %>%
  select(source, rate_pct, n, matched) %>%
  print()

write_json(source_rates, file.path(out_dir, "source_breakdown.json"), pretty = TRUE)

# ── 3. Domain benchmarks ──
# Exclude OSCAR-2019 and OSCAR-2109 (unreliable URL data)
overall_url <- overall_data %>%
  filter(!source %in% c("OSCAR-2019", "OSCAR-2109"))

total_docs <- sum(overall_url$n, na.rm = TRUE)

domain_benchmarks <- tibble(
  domain = c("Government (.gov.cn)", "People's Daily", "Xinhua",
             "Baidu", "Chinese Wikipedia", "Propaganda Match (overall)"),
  docs = c(
    sum(overall_url$n_domain_gov_d, na.rm = TRUE),
    sum(overall_url$n_domain_people_daily, na.rm = TRUE),
    sum(overall_url$n_domain_xinhua, na.rm = TRUE),
    sum(overall_url$n_domain_baidu, na.rm = TRUE),
    sum(overall_url$n_domain_wiki, na.rm = TRUE),
    sum(overall_url$count_2, na.rm = TRUE)
  ),
  total = total_docs
) %>%
  mutate(
    rate = docs / total,
    # For annotation: ratio to Wikipedia
    wiki_rate = rate[domain == "Chinese Wikipedia"],
    ratio_to_wiki = rate / wiki_rate
  ) %>%
  select(-wiki_rate) %>%
  arrange(desc(rate))

cat("\nDomain benchmarks (excl OSCAR-2019, OSCAR-2109):\n")
domain_benchmarks %>%
  mutate(rate_pct = sprintf("%.4f%%", rate * 100)) %>%
  select(domain, rate_pct, docs, ratio_to_wiki) %>%
  print()

write_json(domain_benchmarks, file.path(out_dir, "domain_benchmarks.json"), pretty = TRUE)

# ── 4. Top domains by match count ──
top_domains <- domain_data %>%
  group_by(domain) %>%
  summarize(
    total_docs = sum(n, na.rm = TRUE),
    matched_docs = sum(count_2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(matched_docs > 0) %>%
  mutate(match_rate = matched_docs / total_docs) %>%
  arrange(desc(matched_docs)) %>%
  slice_head(n = 50)

cat("\nTop 10 domains by match count:\n")
top_domains %>%
  slice_head(n = 10) %>%
  mutate(rate_pct = sprintf("%.2f%%", match_rate * 100)) %>%
  select(domain, total_docs, matched_docs, rate_pct) %>%
  print()

write_json(top_domains, file.path(out_dir, "top_domains.json"), pretty = TRUE)

cat("\nDone! JSON files written to", out_dir, "\n")
