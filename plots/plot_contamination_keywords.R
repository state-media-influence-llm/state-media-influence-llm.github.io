#!/usr/bin/env Rscript
# Reproduce the "Match Rates by Keyword" dot plot from contamination.qmd
# Input: data/contamination/keyword_matches.json
# Output: plots/contamination_keywords.pdf

library(tidyverse)
library(jsonlite)
library(scales)
set.seed(42)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

d <- fromJSON(file.path(base_dir, "data/contamination/keyword_matches.json")) %>%
  as_tibble()

cat("Keyword match rates:\n")
d %>% mutate(rate_pct = sprintf("%.2f%%", rate * 100)) %>%
  select(keyword_label, type, rate_pct, n, matched) %>%
  print(n = 20)

type_colors <- c("Leaders" = "#dc3545", "Institutions" = "#2d7bb7", "Not Political" = "#95a5a6")
overall_rate <- d %>% filter(keyword == "weather") %>% pull(rate) %>% first()

p <- d %>%
  mutate(keyword_label = fct_reorder(keyword_label, rate)) %>%
  ggplot(aes(x = rate, y = keyword_label, color = type)) +
  geom_vline(xintercept = overall_rate, linetype = "dashed", color = "#999", linewidth = 0.5) +
  annotate("text", x = overall_rate, y = Inf, label = "1.64% overall",
           vjust = -0.5, hjust = 0.5, color = "#999", size = 3) +
  geom_segment(aes(x = ci_lo, xend = ci_hi, yend = keyword_label), alpha = 0.5, linewidth = 0.8) +
  geom_point(size = 3) +
  scale_color_manual(values = type_colors) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, max(d$ci_hi) * 1.12)) +
  labs(x = "Match rate (% of keyword-tagged documents matching state coordinated media)",
       y = NULL, color = NULL) +
  theme_minimal() +
  theme(panel.grid.major.y = element_line(color = "#eee"),
        panel.grid.minor = element_blank(),
        legend.position = "top")

ggsave(file.path(base_dir, "plots/contamination_keywords.pdf"), p, width = 7, height = 4.5)
cat("Wrote plots/contamination_keywords.pdf\n")
