#!/usr/bin/env Rscript
# Reproduce the "Model Audit" faceted dot plot from audit.qmd
# Input: data/audit/audit_summary.json
# Output: plots/audit_favorability.pdf

library(tidyverse)
library(jsonlite)
library(scales)
set.seed(42)

base_dir <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) dirname(dirname(normalizePath(f, mustWork = FALSE))) else getwd()
}, error = function(e) getwd())

d <- fromJSON(file.path(base_dir, "data/audit/audit_summary.json")) %>%
  as_tibble()

cat("Audit summary:\n")
d %>%
  mutate(est_pct = sprintf("%.1f%%", estimate * 100)) %>%
  select(model, country, facet, est_pct, n, era) %>%
  print(n = 40)

model_colors <- c(
  "Claude Sonnet 3" = "#d6191d",
  "GPT-3.5" = "#fdae61",
  "Claude Opus 3" = "#aad8e9",
  "GPT-4o" = "#2d7bb7",
  "GPT-5.4" = "#1b9e77",
  "Claude Opus 4.6" = "#7570b3",
  "Gemini 3.1 Pro" = "#e7298a",
  "DeepSeek V3.2" = "#e6ab02",
  "Grok 4" = "#a6761d"
)

# Order facets
d <- d %>%
  mutate(facet = factor(facet, levels = c("Baseline", "China", "Spillover")))

# Order models: paper first, then new, alphabetical within each
model_order <- c(
  sort(unique(d$model[d$era == "paper"])),
  sort(unique(d$model[d$era == "new"]))
)
d <- d %>% mutate(model = factor(model, levels = model_order))

p <- d %>%
  ggplot(aes(x = country, y = estimate * 100, color = model)) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "#999") +
  geom_linerange(aes(ymin = lower * 100, ymax = upper * 100),
                 position = position_dodge(width = 0.7), linewidth = 0.6) +
  geom_point(size = 2.5, position = position_dodge(width = 0.7)) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(limits = c(0, 100)) +
  facet_grid(~ facet, scales = "free_x", space = "free_x") +
  labs(y = "% more favorable to Chinese prompt", x = NULL, color = NULL) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face = "bold"))

ggsave(file.path(base_dir, "plots/audit_favorability.pdf"), p, width = 8, height = 4.5)
cat("Wrote plots/audit_favorability.pdf\n")
