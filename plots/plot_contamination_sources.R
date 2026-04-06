#!/usr/bin/env Rscript
# Reproduce "Training Data Sources" + "Domain Composition" from contamination.qmd
# Input: data/contamination/source_breakdown.json, domain_benchmarks.json
# Output: plots/contamination_sources.pdf

library(tidyverse)
library(jsonlite)
library(scales)
library(patchwork)
set.seed(42)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

# --- Panel A: Training Data Sources ---
src <- fromJSON(file.path(base_dir, "data/contamination/source_breakdown.json")) %>%
  as_tibble()

cat("Source breakdown:\n")
src %>% mutate(rate_pct = sprintf("%.3f%%", rate * 100)) %>%
  select(source, rate_pct, n, matched) %>% print()

overall_rate <- src %>% filter(source == "Overall") %>% pull(rate)
src_plot <- src %>% filter(source != "Overall")

p_sources <- src_plot %>%
  mutate(source = fct_reorder(source, rate)) %>%
  ggplot(aes(x = rate, y = source)) +
  geom_vline(xintercept = overall_rate, linetype = "dashed", color = "#999", linewidth = 0.5) +
  annotate("text", x = overall_rate, y = Inf, label = sprintf("%.2f%% overall", overall_rate * 100),
           vjust = -0.5, hjust = 0.5, color = "#999", size = 2.5) +
  geom_segment(aes(x = ci_lo, xend = ci_hi), color = "#2d7bb7", alpha = 0.5, linewidth = 0.8) +
  geom_point(color = "#2d7bb7", size = 3) +
  scale_x_continuous(labels = percent_format(accuracy = 0.1),
                     limits = c(0, max(src_plot$ci_hi) * 1.15)) +
  labs(x = "Match rate", y = NULL, title = "Training Data Sources") +
  theme_minimal() +
  theme(panel.grid.major.y = element_line(color = "#eee"),
        panel.grid.minor = element_blank())

# --- Panel B: Domain Composition ---
dom <- fromJSON(file.path(base_dir, "data/contamination/domain_benchmarks.json")) %>%
  as_tibble()

cat("\nDomain benchmarks:\n")
dom %>% mutate(rate_pct = sprintf("%.4f%%", rate * 100)) %>%
  select(domain, rate_pct, docs, ratio_to_wiki) %>% print()

dom_colors <- case_when(
  dom$domain == "State Coordinated Media Match (overall)" ~ "#dc3545",
  dom$domain == "Chinese Wikipedia" ~ "#2ca02c",
  TRUE ~ "#2d7bb7"
)

p_domains <- dom %>%
  mutate(domain = fct_reorder(domain, rate),
         fill_color = dom_colors) %>%
  ggplot(aes(x = rate, y = domain)) +
  geom_point(aes(color = fill_color), size = 3.5) +
  scale_color_identity() +
  geom_text(data = dom %>% filter(domain == "Government (.gov.cn)"),
            aes(label = "41x Wikipedia"), hjust = -0.15, size = 3, color = "#666", fontface = "italic") +
  scale_x_continuous(labels = percent_format(accuracy = 0.1),
                     limits = c(0, max(dom$rate) * 1.25)) +
  labs(x = "% of CulturaX Chinese-language documents", y = NULL, title = "Domain Composition") +
  theme_minimal() +
  theme(panel.grid.major.y = element_line(color = "#eee"),
        panel.grid.minor = element_blank())

p_combined <- p_sources / p_domains + plot_annotation(tag_levels = "A")
ggsave(file.path(base_dir, "plots/contamination_sources.pdf"), p_combined, width = 7, height = 6)
cat("Wrote plots/contamination_sources.pdf\n")
