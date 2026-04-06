#!/usr/bin/env Rscript
# Reproduce the "Memorization Rate" dot plot from memorization.qmd
# Input: data/memorization/completions.json
# Output: plots/memorization_rates.pdf

library(tidyverse)
library(jsonlite)
library(scales)
set.seed(42)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

raw <- fromJSON(file.path(base_dir, "data/memorization/completions.json"))

# Remap type to match website terminology
raw <- raw %>% mutate(type = if_else(type == "propaganda", "state coordinated media", type))

# Deduplicate: for live models keep latest per (phrase_id, model); paper rows are unique
paper_models <- c("gpt-3.5-instruct", "gpt-4", "gpt-4o", "claude-opus-3", "claude-sonnet-3")

paper <- raw %>% filter(timestamp == "paper")
live <- raw %>%
  filter(timestamp != "paper") %>%
  group_by(phrase_id, model) %>%
  slice_max(timestamp, n = 1, with_ties = FALSE) %>%
  ungroup()
deduped <- bind_rows(paper, live)

# Compute rates per (model, type)
rates <- deduped %>%
  filter(!refused) %>%
  group_by(model, type) %>%
  summarize(
    matched_n = sum(matched, na.rm = TRUE),
    total = n(),
    .groups = "drop"
  ) %>%
  mutate(
    rate = matched_n / total,
    z = 1.96,
    denom_w = 1 + z^2 / total,
    center = (rate + z^2 / (2 * total)) / denom_w,
    margin = z * sqrt((rate * (1 - rate) + z^2 / (4 * total)) / total) / denom_w,
    ci_lo = pmax(0, center - margin),
    ci_hi = pmin(1, center + margin),
    era = if_else(model %in% paper_models, "paper", "new"),
    model_label = if_else(era == "paper", paste0(model, " (paper)"), model)
  ) %>%
  select(-z, -denom_w, -center, -margin)

# Sort by state coordinated media rate descending
scm_order <- rates %>%
  filter(type == "state coordinated media") %>%
  arrange(desc(rate)) %>%
  pull(model_label)

cat("Memorization rates:\n")
rates %>%
  mutate(rate_pct = sprintf("%.1f%%", rate * 100)) %>%
  select(model_label, type, rate_pct, matched_n, total) %>%
  arrange(type, desc(rates$rate)) %>%
  print(n = 30)

type_colors <- c("state coordinated media" = "#dc3545", "culturax" = "#2ca02c")

p <- rates %>%
  mutate(model_label = factor(model_label, levels = scm_order)) %>%
  ggplot(aes(x = rate, y = model_label, color = type)) +
  geom_segment(aes(x = ci_lo, xend = ci_hi, yend = model_label),
               alpha = 0.6, linewidth = 0.8, position = position_dodge(width = 0.5)) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  scale_color_manual(values = type_colors) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, min(1, max(0.15, max(rates$ci_hi)) * 1.15))) +
  labs(x = "Memorization rate", y = NULL, color = NULL) +
  theme_minimal() +
  theme(panel.grid.major.y = element_line(color = "#eee"),
        panel.grid.minor = element_blank(),
        legend.position = "top")

ggsave(file.path(base_dir, "plots/memorization_rates.pdf"), p, width = 7, height = 6)
cat("Wrote plots/memorization_rates.pdf\n")
